
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch import nn, Tensor

from src.diffusion.schedule import (
    DDPMSchedule,
)
from src.utils.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def load_overfit_config(
    path: Path,
) -> dict:
    # Carica la configurazione dedicata al tiny overfit
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "overfit" not in config:
        raise ValueError(
            "Sezione overfit mancante."
        )

    return config


def set_training_seed(
    seed: int,
) -> None:
    # Rende il training riproducibile
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TinyOverfitTrainer:

    def __init__(
        self,
        conditional_unet: nn.Module,
        text_encoder: nn.Module,
        schedule: DDPMSchedule,
        optimizer: torch.optim.Optimizer,
        fixed_batch: dict,
        device: torch.device,
        config: dict,
    ) -> None:
        # Il tiny overfit è pensato per GPU
        if device.type != "cuda":
            raise ValueError(
                "Il tiny-overfit richiede CUDA."
            )

        self.conditional_unet = (
            conditional_unet
        )
        self.text_encoder = text_encoder
        self.schedule = schedule
        self.optimizer = optimizer
        self.fixed_batch = fixed_batch
        self.device = device
        self.config = config

        overfit_config = config["overfit"]

        self.gradient_clip_norm = float(
            overfit_config[
                "gradient_clip_norm"
            ]
        )

        self.amp_enabled = bool(
            overfit_config["amp"]["enabled"]
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

        self.train_generator = (
            torch.Generator(
                device=device
            ).manual_seed(
                int(overfit_config["seed"])
            )
        )

        self.evaluation_seed = int(
            overfit_config["evaluation"][
                "seed"
            ]
        )

        self.step = 0
        self.best_evaluation_loss = math.inf
        self.history = []

        self.parameters = [
            *self.conditional_unet.parameters(),
            *self.text_encoder.parameters(),
        ]

    def _prepare_batch(
        self,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Sposta il batch fisso sul device di training
        images = self.fixed_batch[
            "image"
        ].to(
            self.device,
            non_blocking=True,
        )

        input_ids = self.fixed_batch[
            "input_ids"
        ].to(
            self.device,
            non_blocking=True,
        )

        attention_mask = self.fixed_batch[
            "attention_mask"
        ].to(
            self.device,
            non_blocking=True,
        )

        return (
            images,
            input_ids,
            attention_mask,
        )

    def _build_noisy_batch(
        self,
        images: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Campiona i timestep e costruisce x_t con il rumore target
        timesteps = torch.randint(
            low=0,
            high=self.schedule.num_timesteps,
            size=(images.shape[0],),
            device=self.device,
            dtype=torch.long,
            generator=generator,
        )

        noisyimages, target_noise = (
            self.schedule.q_sample(
                x_start=images,
                timesteps=timesteps,
                generator=generator,
            )
        )

        return (
            noisyimages,
            target_noise,
            timesteps,
        )

    def _predict_noise(
        self,
        noisyimages,
        timesteps,
        input_ids,
        attention_mask,
    ):
        # Verifica minima sulla correttezza del conditioning
        if attention_mask is None:
            raise ValueError(
                "_predict_noise ha ricevuto attention_mask=None."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask deve avere shape "
                "[batch, sequence_length], ma ha "
                f"shape {tuple(attention_mask.shape)}."
            )

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids e attention_mask devono avere "
                "la stessa shape: "
                f"{tuple(input_ids.shape)} contro "
                f"{tuple(attention_mask.shape)}."
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

    def train_step(
        self,
    ) -> dict:
        # Esegue un singolo passo di training sul batch fisso
        self.conditional_unet.train()
        self.text_encoder.train()

        (
            images,
            input_ids,
            attention_mask,
        ) = self._prepare_batch()

        (
            noisyimages,
            target_noise,
            timesteps,
        ) = self._build_noisy_batch(
            images=images,
            generator=self.train_generator,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            predicted_noise = (
                self._predict_noise(
                    noisyimages=noisyimages,
                    timesteps=timesteps,
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                )
            )

            loss = F.mse_loss(
                predicted_noise.float(),
                target_noise.float(),
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss non finita allo step "
                f"{self.step}: {loss.item()}"
            )

        self.scaler.scale(
            loss
        ).backward()

        self.scaler.unscale_(
            self.optimizer
        )

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.parameters,
                max_norm=(
                    self.gradient_clip_norm
                ),
                error_if_nonfinite=True,
            )
        )

        self.scaler.step(
            self.optimizer
        )
        self.scaler.update()

        self.step += 1

        return {
            "step": self.step,
            "train_loss": float(
                loss.detach().item()
            ),
            "gradient_norm": float(
                gradient_norm.detach().item()
            ),
            "amp_scale": float(
                self.scaler.get_scale()
            ),
        }

    @torch.inference_mode()
    def evaluate(
        self,
    ) -> float:
        # Valutazione deterministica con seed fisso
        self.conditional_unet.eval()
        self.text_encoder.eval()

        (
            images,
            input_ids,
            attention_mask,
        ) = self._prepare_batch()

        generator = (
            torch.Generator(
                device=self.device
            ).manual_seed(
                self.evaluation_seed
            )
        )

        (
            noisyimages,
            target_noise,
            timesteps,
        ) = self._build_noisy_batch(
            images=images,
            generator=generator,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            predicted_noise = (
                self._predict_noise(
                    noisyimages=noisyimages,
                    timesteps=timesteps,
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                )
            )

            loss = F.mse_loss(
                predicted_noise.float(),
                target_noise.float(),
            )

        return float(loss.item())

    def save(
        self,
        path: Path,
    ) -> None:
        # Salva lo stato completo del tiny overfit
        save_training_checkpoint(
            path=path,
            models={
                "conditional_unet": (
                    self.conditional_unet
                ),
                "text_encoder": (
                    self.text_encoder
                ),
            },
            optimizer=self.optimizer,
            scaler=self.scaler,
            step=self.step,
            metadata={
                "best_evaluation_loss": (
                    self.best_evaluation_loss
                ),
                "history": self.history,
            },
            configs=self.config,
            generators={
                "training": (
                    self.train_generator
                )
            },
        )

    def load(
        self,
        path: Path,
    ) -> dict:
        # Ripristina un run precedente
        checkpoint = (
            load_training_checkpoint(
                path=path,
                models={
                    "conditional_unet": (
                        self.conditional_unet
                    ),
                    "text_encoder": (
                        self.text_encoder
                    ),
                },
                optimizer=self.optimizer,
                scaler=self.scaler,
                generators={
                    "training": (
                        self.train_generator
                    )
                },
                map_location=self.device,
                restore_rng=True,
            )
        )

        self.step = int(
            checkpoint["step"]
        )

        metadata = checkpoint[
            "metadata"
        ]

        self.best_evaluation_loss = float(
            metadata.get(
                "best_evaluation_loss",
                math.inf,
            )
        )

        self.history = list(
            metadata.get(
                "history",
                [],
            )
        )

        return checkpoint

    def write_history(
        self,
        path: Path,
    ) -> None:
        # Esporta la history su CSV
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.history:
            return

        keys = [
            "step",
            "train_loss",
            "evaluation_loss",
            "gradient_norm",
            "amp_scale",
        ]

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=keys,
            )

            writer.writeheader()

            for row in self.history:
                writer.writerow(
                    {
                        key: row.get(
                            key,
                            "",
                        )
                        for key in keys
                    }
                )

    def fit(
        self,
        max_steps: int,
        evaluation_every: int,
        checkpoint_every: int,
        checkpoint_directory: Path,
        history_path: Path,
        logging_every: int,
    ) -> None:
        # Loop principale di tiny overfit
        checkpoint_directory = Path(
            checkpoint_directory
        )

        checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        latest_path = (
            checkpoint_directory
            / "latest.pt"
        )

        best_path = (
            checkpoint_directory
            / "best.pt"
        )

        while self.step < max_steps:
            metrics = self.train_step()

            evaluation_loss = None

            if (
                self.step == 1
                or self.step
                % evaluation_every == 0
            ):
                evaluation_loss = (
                    self.evaluate()
                )

                metrics[
                    "evaluation_loss"
                ] = evaluation_loss

                if (
                    evaluation_loss
                    < self.best_evaluation_loss
                ):
                    self.best_evaluation_loss = (
                        evaluation_loss
                    )

                    self.history.append(
                        metrics.copy()
                    )

                    self.save(
                        best_path
                    )

                    self.history.pop()

            self.history.append(metrics)

            if (
                self.step == 1
                or self.step
                % logging_every == 0
            ):
                print(
                    f"step={self.step:05d} "
                    f"train={metrics['train_loss']:.6f} "
                    f"eval="
                    f"{evaluation_loss if evaluation_loss is not None else '—'} "
                    f"grad={metrics['gradient_norm']:.4f} "
                    f"scale={metrics['amp_scale']:.1f}"
                )

            if (
                self.step
                % checkpoint_every == 0
                or self.step == max_steps
            ):
                self.save(
                    latest_path
                )

            self.write_history(
                history_path
            )

        print(
            "Training terminato. "
            f"Step={self.step}, "
            f"best_eval="
            f"{self.best_evaluation_loss:.6f}"
        )
