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
        schedule: DDPMSchedule,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
        config: dict,
        text_encoder: Optional[nn.Module] = None,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.conditional_unet = conditional_unet
        self.text_encoder = text_encoder
        self.schedule = schedule
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.config = config

        training_config = config["training"]

        # Configurazione numerica del training
        self.gradient_clip_norm = float(
            training_config.get("gradient_clip_norm", 1.0)
        )

        # AMP attivo solo quando richiesto e disponibile
        requested_amp = bool(
            training_config.get("amp", {}).get("enabled", False)
        )

        self.device_type = self.device.type
        self.amp_enabled = requested_amp and self.device_type == "cuda"
        self.amp_dtype = torch.float16

        self.scaler = torch.amp.GradScaler(
            self.device_type,
            enabled=self.amp_enabled,
        )

        seed = int(training_config.get("seed", 42))
        self.train_generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        self.evaluation_seed = int(
            training_config.get("evaluation", {}).get("seed", 3107)
        )

        self.step = 0
        self.epoch = 0
        self.best_evaluation_loss = math.inf
        self.history: list[dict] = []

        self.parameters = list(self.conditional_unet.parameters())

        # Il trainer è conditional se è presente il text encoder
        if self.text_encoder is not None:
            self.parameters.extend(self.text_encoder.parameters())

    @property
    def is_conditional(self) -> bool:
        return self.text_encoder is not None

    def prepare_batch(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            self.device,
            non_blocking=True,
        )

        return images, input_ids, attention_mask

    def build_noisy_batch(
        self,
        images: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        timesteps = torch.randint(
            low=0,
            high=self.schedule.num_timesteps,
            size=(images.shape[0],),
            device=self.device,
            dtype=torch.long,
            generator=generator,
        )

        noisyimages, target_noise = self.schedule.q_sample(
            x_start=images,
            timesteps=timesteps,
            generator=generator,
        )

        return noisyimages, target_noise, timesteps

    def predict_noise(
        self,
        noisyimages: Tensor,
        timesteps: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        # In unconditional la U-Net viene invocata senza context
        if not self.is_conditional:
            return self.conditional_unet(
                noisyimages=noisyimages,
                timesteps=timesteps,
                context=None,
                attention_mask=None,
            )

        text_context = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.conditional_unet(
            noisyimages=noisyimages,
            timesteps=timesteps,
            context=text_context,
            attention_mask=attention_mask,
        )

    def train_step(self, batch: dict) -> dict:
        self.conditional_unet.train()

        if self.text_encoder is not None:
            self.text_encoder.train()

        images, input_ids, attention_mask = self.prepare_batch(batch)

        noisyimages, target_noise, timesteps = self.build_noisy_batch(
            images=images,
            generator=self.train_generator,
        )

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            predicted_noise = self.predict_noise(
                noisyimages=noisyimages,
                timesteps=timesteps,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = F.mse_loss(
                predicted_noise.float(),
                target_noise.float(),
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss non finita allo step {self.step}: {loss.item()}"
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters,
            max_norm=self.gradient_clip_norm,
            error_if_nonfinite=True,
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
        if self.validation_dataloader is None:
            raise RuntimeError(
                "evaluate() chiamata senza validation_dataloader."
            )

        self.conditional_unet.eval()

        if self.text_encoder is not None:
            self.text_encoder.eval()

        total_loss = 0.0
        batches = 0

        generator = torch.Generator(
            device=self.device
        ).manual_seed(self.evaluation_seed)

        for batch in self.validation_dataloader:
            images, input_ids, attention_mask = self.prepare_batch(batch)

            noisyimages, target_noise, timesteps = self.build_noisy_batch(
                images=images,
                generator=generator,
            )

            with torch.autocast(
                device_type=self.device_type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                predicted_noise = self.predict_noise(
                    noisyimages=noisyimages,
                    timesteps=timesteps,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = F.mse_loss(
                    predicted_noise.float(),
                    target_noise.float(),
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Loss non finita durante evaluation."
                )

            total_loss += float(loss.item())
            batches += 1

        # La validation loss è mediata su tutto il validation set
        return total_loss / max(1, batches)

    def save(self, path: Path) -> None:
        models = {"conditional_unet": self.conditional_unet}

        if self.text_encoder is not None:
            models["text_encoder"] = self.text_encoder

        save_training_checkpoint(
            path=path,
            models=models,
            optimizer=self.optimizer,
            scaler=self.scaler,
            step=self.step,
            epoch=self.epoch,
            metadata={
                "best_evaluation_loss": self.best_evaluation_loss,
                "history": self.history,
                "conditional": self.is_conditional,
            },
            configs=self.config,
            generators={"training": self.train_generator},
        )

    def load(self, path: Path) -> None:
        models = {"conditional_unet": self.conditional_unet}

        if self.text_encoder is not None:
            models["text_encoder"] = self.text_encoder

        checkpoint = load_training_checkpoint(
            path=path,
            models=models,
            optimizer=self.optimizer,
            scaler=self.scaler,
            map_location=self.device,
            generators={"training": self.train_generator},
        )

        self.step = int(checkpoint["step"])
        self.epoch = int(checkpoint["epoch"])

        metadata = checkpoint.get("metadata", {})
        self.best_evaluation_loss = float(
            metadata.get("best_evaluation_loss", math.inf)
        )
        self.history = list(metadata.get("history", []))

    def write_history(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        keys = [
            "step",
            "epoch",
            "train_loss",
            "evaluation_loss",
            "gradient_norm",
            "amp_scale",
        ]

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()

            for row in self.history:
                writer.writerow({key: row.get(key) for key in keys})

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

        train_iterator = iter(self.train_dataloader)

        while self.step < max_steps:
            try:
                batch = next(train_iterator)

            except StopIteration:
                self.epoch += 1
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            metrics = self.train_step(batch)
            evaluation_loss = None

            should_evaluate = (
                self.validation_dataloader is not None
                and evaluation_every > 0
                and (self.step == 1 or self.step % evaluation_every == 0)
            )

            if should_evaluate:
                evaluation_loss = self.evaluate()
                metrics["evaluation_loss"] = evaluation_loss

                if evaluation_loss < self.best_evaluation_loss:
                    self.best_evaluation_loss = evaluation_loss
                    self.history.append(metrics.copy())
                    self.save(best_path)
                    self.history.pop()

            self.history.append(metrics)

            if self.step == 1 or self.step % logging_every == 0:
                evaluation_string = (
                    f"{evaluation_loss:.6f}"
                    if evaluation_loss is not None
                    else "-"
                )

                print(
                    f"Epoch {self.epoch} "
                    f"Step {self.step:05d} "
                    f"Train {metrics['train_loss']:.6f} "
                    f"Eval {evaluation_string} "
                    f"Grad {metrics['gradient_norm']:.4f}"
                )

            # latest.pt salva lo stato corrente, best.pt il miglior modello
            if self.step % checkpoint_every == 0 or self.step == max_steps:
                self.save(latest_path)
                self.write_history(history_path)

        print(
            f"Training terminato. "
            f"Step={self.step}, "
            f"Epoch={self.epoch}, "
            f"Best eval={self.best_evaluation_loss:.6f}"
        )
