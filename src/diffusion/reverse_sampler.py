from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn

from src.diffusion.schedule import DDPMSchedule


def load_sampling_config(
    config_path: Path,
) -> dict:
    with Path(config_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "sampling" not in config:
        raise ValueError(
            "Sezione 'sampling' mancante."
        )

    sampling_config = config["sampling"]

    required_keys = {
        "guidance_scale",
        "clip_denoised",
    }

    missing_keys = required_keys - set(sampling_config)

    if missing_keys:
        raise ValueError(
            "Parametri sampling mancanti: "
            f"{sorted(missing_keys)}"
        )

    guidance_scale = float(
        sampling_config["guidance_scale"]
    )

    if guidance_scale < 0.0:
        raise ValueError(
            "guidance_scale non può essere negativo."
        )

    if not isinstance(
        sampling_config["clip_denoised"],
        bool,
    ):
        raise TypeError(
            "clip_denoised deve essere booleano."
        )

    return config


class ReverseSampler:
    # Sampler DDPM condizionato dal testo.

    # Responsabilità:
    # - costruire il contesto testuale condizionato;
    # - costruire il contesto unconditional per CFG;
    # - interrogare la U-Net ad ogni timestep;
    # - delegare alla schedule il reverse step DDPM.

    def __init__(
        self,
        conditional_unet: nn.Module,
        text_encoder: nn.Module,
        schedule: DDPMSchedule,
        guidance_scale: float = 1.0,
        clip_denoised: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.conditional_unet = conditional_unet
        self.text_encoder = text_encoder
        self.schedule = schedule

        self.guidance_scale = float(guidance_scale)
        self.clip_denoised = bool(clip_denoised)

        if self.guidance_scale < 0.0:
            raise ValueError(
                "guidance_scale non può essere negativo."
            )

        if device is None:
            try:
                device = next(
                    conditional_unet.parameters()
                ).device
            except StopIteration:
                device = torch.device("cpu")

        self.device = torch.device(device)

        self.conditional_unet.to(self.device)
        self.text_encoder.to(self.device)
        self.schedule.to(self.device)

        pad_token_id = getattr(
            self.text_encoder,
            "pad_token_id",
            None,
        )

        if pad_token_id is None:
            raise AttributeError(
                "Il text encoder deve esporre "
                "pad_token_id."
            )

        self.pad_token_id = int(pad_token_id)

    @classmethod
    def from_config(
        cls,
        conditional_unet: nn.Module,
        text_encoder: nn.Module,
        schedule: DDPMSchedule,
        config: dict,
        device: Optional[torch.device] = None,
    ) -> "ReverseSampler":
        sampling_config = config["sampling"]

        return cls(
            conditional_unet=conditional_unet,
            text_encoder=text_encoder,
            schedule=schedule,
            guidance_scale=float(
                sampling_config["guidance_scale"]
            ),
            clip_denoised=bool(
                sampling_config["clip_denoised"]
            ),
            device=device,
        )

    def _validate_tokens(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids deve avere shape "
                "[batch, sequence_length]."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask deve avere shape "
                "[batch, sequence_length]."
            )

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids e attention_mask devono "
                "avere la stessa shape."
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids deve avere dtype torch.long."
            )

        if attention_mask.dtype != torch.long:
            raise TypeError(
                "attention_mask deve avere dtype torch.long."
            )

    def _build_unconditional_inputs(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Costruisce la condizione nulla per la CFG:
        # tutti PAD, mask a zero.
        unconditional_ids = torch.full_like(
            input_ids,
            fill_value=self.pad_token_id,
        )

        unconditional_mask = torch.zeros_like(
            input_ids,
            dtype=torch.long,
        )

        return unconditional_ids, unconditional_mask

    def _predict_noise(
        self,
        noisyimages: Tensor,
        timesteps: Tensor,
        conditional_context: Tensor,
        conditional_mask: Tensor,
        unconditional_context: Optional[Tensor] = None,
        unconditional_mask: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        conditional_noise = self.conditional_unet(
            noisyimages=noisyimages,
            timesteps=timesteps,
            context=conditional_context,
            attention_mask=conditional_mask,
        )

        if guidance_scale == 1.0:
            return conditional_noise

        if (
            unconditional_context is None
            or unconditional_mask is None
        ):
            raise ValueError(
                "CFG richiede context e mask unconditional."
            )

        unconditional_noise = self.conditional_unet(
            noisyimages=noisyimages,
            timesteps=timesteps,
            context=unconditional_context,
            attention_mask=unconditional_mask,
        )

        return unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )

    def _p_sample(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        # Deleghiamo il reverse step alla schedule per avere
        # una sola implementazione matematica condivisa.
        return self.schedule.reverse_step(
            noisyimages=x_t,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
            generator=generator,
            clip_denoised=self.clip_denoised,
        )

    @torch.inference_mode()
    def sample(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        image_shape: tuple[int, int, int],
        generator: Optional[torch.Generator] = None,
        guidance_scale: Optional[float] = None,
        return_intermediates: bool = False,
        record_every: Optional[int] = None,
    ):
        self._validate_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if len(image_shape) != 3:
            raise ValueError(
                "image_shape deve essere "
                "(channels, height, width)."
            )

        channels, height, width = image_shape

        if min(channels, height, width) < 1:
            raise ValueError(
                "Le dimensioni immagine devono essere positive."
            )

        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(
                "Altezza e larghezza devono essere "
                "divisibili per 4."
            )

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        guidance_scale = float(guidance_scale)

        if guidance_scale < 0.0:
            raise ValueError(
                "guidance_scale non può essere negativo."
            )

        input_ids = input_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        attention_mask = attention_mask.to(
            device=self.device,
            dtype=torch.long,
        )

        batch_size = input_ids.shape[0]

        self.conditional_unet.eval()
        self.text_encoder.eval()

        conditional_context = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        unconditional_ids, unconditional_mask = (
            self._build_unconditional_inputs(
                input_ids=input_ids
            )
        )

        unconditional_context = self.text_encoder(
            input_ids=unconditional_ids,
            attention_mask=unconditional_mask,
        )

        x_t = torch.randn(
            batch_size,
            channels,
            height,
            width,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        intermediates = []

        for timestep in reversed(
            range(self.schedule.num_timesteps)
        ):
            timesteps = torch.full(
                (batch_size,),
                fill_value=timestep,
                device=self.device,
                dtype=torch.long,
            )

            predicted_noise = self._predict_noise(
                noisyimages=x_t,
                timesteps=timesteps,
                conditional_context=conditional_context,
                conditional_mask=attention_mask,
                unconditional_context=unconditional_context,
                unconditional_mask=unconditional_mask,
                guidance_scale=guidance_scale,
            )

            x_t, predicted_x_start = self._p_sample(
                x_t=x_t,
                timesteps=timesteps,
                predicted_noise=predicted_noise,
                generator=generator,
            )

            should_record = (
                return_intermediates
                and (
                    record_every is None
                    or timestep
                    == self.schedule.num_timesteps - 1
                    or timestep == 0
                    or timestep % record_every == 0
                )
            )

            if should_record:
                intermediates.append(
                    (
                        timestep,
                        predicted_x_start.detach().cpu(),
                    )
                )

        x_t = x_t.clamp(min=-1.0, max=1.0)

        if return_intermediates:
            return x_t, intermediates

        return x_t
