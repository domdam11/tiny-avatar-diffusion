
from collections.abc import Callable
from typing import Optional

import torch
from torch import Tensor

from src.diffusion.schedule import (
    DDPMSchedule,
)


NoisePredictor = Callable[
    [Tensor, Tensor],
    Tensor,
]


def _randn_like(
    reference: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def p_mean_variance(
    schedule: DDPMSchedule,
    x_t: Tensor,
    timesteps: Tensor,
    predicted_noise: Tensor,
    clip_denoised: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if x_t.shape != predicted_noise.shape:
        raise ValueError(
            "x_t e predicted_noise devono avere "
            "la stessa shape."
        )

    predicted_x_start = (
        schedule.predict_start_from_noise(
            x_t=x_t,
            timesteps=timesteps,
            noise=predicted_noise,
        )
    )

    if clip_denoised:
        predicted_x_start = (
            predicted_x_start.clamp(
                min=-1.0,
                max=1.0,
            )
        )

    (
        posterior_mean,
        posterior_variance,
        posterior_log_variance,
    ) = schedule.q_posterior(
        x_start=predicted_x_start,
        x_t=x_t,
        timesteps=timesteps,
    )

    return (
        posterior_mean,
        posterior_variance,
        posterior_log_variance,
        predicted_x_start,
    )


def p_sample(
    schedule: DDPMSchedule,
    noise_predictor: NoisePredictor,
    x_t: Tensor,
    timesteps: Tensor,
    generator: Optional[
        torch.Generator
    ] = None,
    noise: Optional[Tensor] = None,
    clip_denoised: bool = True,
) -> tuple[Tensor, Tensor]:
    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps deve avere shape [batch]."
        )

    if timesteps.shape[0] != x_t.shape[0]:
        raise ValueError(
            "Batch di timestep e immagini diversi."
        )

    if noise is not None and generator is not None:
        raise ValueError(
            "Specificare noise oppure generator, "
            "non entrambi."
        )

    predicted_noise = noise_predictor(
        x_t,
        timesteps,
    )

    if predicted_noise.shape != x_t.shape:
        raise ValueError(
            "Il noise predictor ha restituito "
            f"{tuple(predicted_noise.shape)}, "
            f"attesa {tuple(x_t.shape)}."
        )

    (
        posterior_mean,
        posterior_variance,
        _,
        predicted_x_start,
    ) = p_mean_variance(
        schedule=schedule,
        x_t=x_t,
        timesteps=timesteps,
        predicted_noise=predicted_noise,
        clip_denoised=clip_denoised,
    )

    if noise is None:
        noise = _randn_like(
            reference=x_t,
            generator=generator,
        )

    if noise.shape != x_t.shape:
        raise ValueError(
            "noise e x_t devono avere "
            "la stessa shape."
        )

    nonzero_mask = (
        timesteps.ne(0)
        .to(dtype=x_t.dtype)
        .reshape(
            x_t.shape[0],
            *([1] * (x_t.ndim - 1)),
        )
    )

    x_previous = (
        posterior_mean
        + nonzero_mask
        * torch.sqrt(
            posterior_variance
        )
        * noise
    )

    return (
        x_previous,
        predicted_x_start,
    )


@torch.inference_mode()
def sample_loop(
    schedule: DDPMSchedule,
    noise_predictor: NoisePredictor,
    shape: tuple[int, int, int, int],
    device: torch.device,
    generator: Optional[
        torch.Generator
    ] = None,
    initial_noise: Optional[Tensor] = None,
    clip_denoised: bool = True,
    record_every: Optional[int] = None,
) -> tuple[Tensor, list[tuple[int, Tensor]]]:
    if len(shape) != 4:
        raise ValueError(
            "shape deve essere [B, C, H, W]."
        )

    if shape[0] < 1:
        raise ValueError(
            "Il batch deve contenere almeno "
            "un elemento."
        )

    if initial_noise is None:
        x_t = torch.randn(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(
                "Shape del rumore iniziale errata."
            )

        x_t = initial_noise.to(
            device=device,
            dtype=torch.float32,
        ).clone()

    intermediates = []

    for timestep in reversed(
        range(schedule.num_timesteps)
    ):
        timesteps = torch.full(
            size=(shape[0],),
            fill_value=timestep,
            device=device,
            dtype=torch.long,
        )

        x_t, predicted_x_start = p_sample(
            schedule=schedule,
            noise_predictor=noise_predictor,
            x_t=x_t,
            timesteps=timesteps,
            generator=generator,
            clip_denoised=clip_denoised,
        )

        should_record = (
            record_every is not None
            and (
                timestep % record_every == 0
                or timestep
                == schedule.num_timesteps - 1
                or timestep == 0
            )
        )

        if should_record:
            intermediates.append(
                (
                    timestep,
                    predicted_x_start
                    .detach()
                    .cpu(),
                )
            )

    return (
        x_t.clamp(-1.0, 1.0),
        intermediates,
    )
