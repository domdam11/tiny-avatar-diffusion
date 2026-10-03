
import csv
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.diffusion.schedule import DDPMSchedule
from src.utils.checkpoint import load_training_checkpoint, save_training_checkpoint


def load_training_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "training" not in config:
        raise ValueError("Sezione 'training' mancante nel file di configurazione.")

    return config


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(
        self,
        conditional_unet: nn.Module,
        text_encoder: nn.Module,
        schedule: DDPMSchedule,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
        config: dict,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.conditional_unet = conditional_unet
        self.text_encoder = text_encoder
        self.schedule = schedule
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config

        training_config = config["training"]
        self.gradient_clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
        self.amp_enabled = bool(training_config.get("amp", {}).get("enabled", False))

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        seed = int(training_config.get("seed", 42))
        self.train_generator = torch.Generator(device=device).manual_seed(seed)
        self.evaluation_seed = int(training_config.get("evaluation", {}).get("seed", 3107))

        self.step = 0
        self.epoch = 0
        self.best_evaluation_loss = math.inf
        self.history = []

        self.parameters = [
            *self.conditional_unet.parameters(),
            *self.text_encoder.parameters(),
        ]

    def _prepare_batch(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        return images, input_ids, attention_mask

    def _build_noisy_batch(
        self, images: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor, Tensor]:
        timesteps = torch.randint(
            low=0,
            high=self.schedule.num_timesteps,
            size=(images.shape[0],),
            device=self.device,
            dtype=torch.long,
            generator=generator,
        )
        noisy_images, target_noise = self.schedule.q_sample(
            x_start=images, timesteps=timesteps, generator=generator
        )
        return noisy_images, target_noise, timesteps

    def _predict_noise(
        self, noisy_images: Tensor, timesteps: Tensor, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        text_context = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.conditional_unet(
            noisy_images=noisy_images,
            timesteps=timesteps,
            context=text_context,
            attention_mask=attention_mask,
        )

    def train_step(self, batch: dict) -> dict:
        self.conditional_unet.train()
        self.text_encoder.train()

        images, input_ids, attention_mask = self._prepare_batch(batch)
        noisy_images, target_noise, timesteps = self._build_noisy_batch(
            images=images, generator=self.train_generator
        )

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled):
            predicted_noise = self._predict_noise(
                noisy_images=noisy_images,
                timesteps=timesteps,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = F.mse_loss(predicted_noise.float(), target_noise.float())

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss non finita allo step {self.step}: {loss.item()}")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, max_norm=self.gradient_clip_norm, error_if_nonfinite=True
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1

        return {
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "amp_scale": float(self.scaler.get_scale()),
        }

    @torch.inference_mode()
    def evaluate(self) -> float:
        if not self.val_dataloader:
            return 0.0

        self.conditional_unet.eval()
        self.text_encoder.eval()

        total_loss = 0.0
        batches = 0
        generator = torch.Generator(device=self.device).manual_seed(self.evaluation_seed)

        for batch in self.val_dataloader:
            images, input_ids, attention_mask = self._prepare_batch(batch)
            noisy_images, target_noise, timesteps = self._build_noisy_batch(
                images=images, generator=generator
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled):
                predicted_noise = self._predict_noise(
                    noisy_images=noisy_images,
                    timesteps=timesteps,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = F.mse_loss(predicted_noise.float(), target_noise.float())

            total_loss += float(loss.item())
            batches += 1

        return total_loss / max(1, batches)

    def save(self, path: Path) -> None:
        save_training_checkpoint(
            path=path,
            models={"conditional_unet": self.conditional_unet, "text_encoder": self.text_encoder},
            optimizer=self.optimizer,
            scaler=self.scaler,
            step=self.step,
            epoch=self.epoch,
            metadata={"best_evaluation_loss": self.best_evaluation_loss, "history": self.history},
            configs=self.config,
            generators={"training": self.train_generator},
        )

    def write_history(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            return

        keys = ["step", "epoch", "train_loss", "evaluation_loss", "gradient_norm", "amp_scale"]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow({key: row.get(key, "") for key in keys})

    def fit(
        self,
        max_steps: int,
        evaluation_every: int,
        checkpoint_every: int,
        checkpoint_directory: Path,
        history_path: Path,
        logging_every: int,
    ) -> None:
        checkpoint_directory = Path(checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_directory / "latest.pt"
        best_path = checkpoint_directory / "best.pt"

        train_iter = iter(self.train_dataloader)

        while self.step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            metrics = self.train_step(batch)
            evaluation_loss = None

            if self.step == 1 or self.step % evaluation_every == 0:
                evaluation_loss = self.evaluate()
                metrics["evaluation_loss"] = evaluation_loss

                if evaluation_loss < self.best_evaluation_loss and self.val_dataloader:
                    self.best_evaluation_loss = evaluation_loss
                    self.history.append(metrics.copy())
                    self.save(best_path)
                    self.history.pop()

            self.history.append(metrics)

            if self.step == 1 or self.step % logging_every == 0:
                eval_str = f"{evaluation_loss:.6f}" if evaluation_loss is not None else "—"
                print(
                    f"Epoch={self.epoch} Step={self.step:05d} "
                    f"Train={metrics['train_loss']:.6f} Eval={eval_str} "
                    f"Grad={metrics['gradient_norm']:.4f}"
                )

            if self.step % checkpoint_every == 0 or self.step == max_steps:
                self.save(latest_path)
                self.write_history(history_path)

        print(f"Training terminato. Step={self.step}, Epoch={self.epoch}, Best Eval={self.best_evaluation_loss:.6f}")
