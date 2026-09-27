
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
import yaml


def load_diffusion_config(
    config_path: Path,
) -> dict:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "diffusion" not in config:
        raise ValueError(
            "Sezione 'diffusion' mancante."
        )

    if "data" not in config:
        raise ValueError(
            "Sezione 'data' mancante."
        )

    diffusion = config["diffusion"]
    data = config["data"]

    required_diffusion = {
        "num_timesteps",
        "beta_schedule",
        "prediction_type",
    }

    missing = (
        required_diffusion
        - set(diffusion)
    )

    if missing:
        raise ValueError(
            "Parametri diffusion mancanti: "
            f"{sorted(missing)}"
        )

    schedule = diffusion["beta_schedule"]

    if schedule.get("type") != "linear":
        raise ValueError(
            "In questa implementazione iniziale "
            "è supportato solo lo schedule lineare."
        )

    beta_start = float(
        schedule["beta_start"]
    )
    beta_end = float(
        schedule["beta_end"]
    )

    num_timesteps = int(
        diffusion["num_timesteps"]
    )

    if num_timesteps < 2:
        raise ValueError(
            "num_timesteps deve essere almeno 2."
        )

    if not (
        0.0
        < beta_start
        < beta_end
        < 1.0
    ):
        raise ValueError(
            "È richiesto "
            "0 < beta_start < beta_end < 1."
        )

    if diffusion["prediction_type"] != "epsilon":
        raise ValueError(
            "Il progetto usa epsilon prediction."
        )

    if int(data["image_size"]) != 32:
        raise ValueError(
            "La configurazione iniziale richiede "
            "immagini 32x32."
        )

    if int(data["image_channels"]) != 3:
        raise ValueError(
            "Il dataset deve essere RGB."
        )

    normalization = data["normalization"]

    if (
        float(normalization["minimum"]) != -1.0
        or float(normalization["maximum"]) != 1.0
    ):
        raise ValueError(
            "Le immagini devono essere "
            "normalizzate in [-1, 1]."
        )

    return config


def extract(
    values: Tensor,
    timesteps: Tensor,
    reference: Tensor,
) -> Tensor:
    if timesteps.dtype != torch.long:
        raise TypeError(
            "timesteps deve avere dtype torch.long."
        )

    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps deve avere shape [batch]."
        )

    if reference.ndim < 2:
        raise ValueError(
            "reference deve avere almeno 2 dimensioni."
        )

    if (
        timesteps.shape[0]
        != reference.shape[0]
    ):
        raise ValueError(
            "La dimensione batch dei timestep "
            "non coincide con quella del tensore."
        )

    if values.device != reference.device:
        raise ValueError(
            "Schedule e tensore sono su device "
            "diversi. Eseguire schedule.to(device)."
        )

    if timesteps.device != reference.device:
        raise ValueError(
            "Timestep e tensore sono su device diversi."
        )

    if timesteps.numel() > 0:
        minimum = int(
            timesteps.min().item()
        )
        maximum = int(
            timesteps.max().item()
        )

        if minimum < 0:
            raise IndexError(
                "I timestep non possono essere negativi."
            )

        if maximum >= values.shape[0]:
            raise IndexError(
                f"Timestep {maximum} fuori intervallo. "
                f"Massimo consentito: "
                f"{values.shape[0] - 1}."
            )

    gathered = values.gather(
        dim=0,
        index=timesteps,
    )

    output_shape = (
        timesteps.shape[0],
        *([1] * (reference.ndim - 1)),
    )

    return (
        gathered
        .reshape(output_shape)
        .to(dtype=reference.dtype)
    )


class DDPMSchedule(nn.Module):

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()

        if num_timesteps < 2:
            raise ValueError(
                "num_timesteps deve essere almeno 2."
            )

        if not (
            0.0
            < beta_start
            < beta_end
            < 1.0
        ):
            raise ValueError(
                "È richiesto "
                "0 < beta_start < beta_end < 1."
            )

        self.num_timesteps = int(
            num_timesteps
        )
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        # I coefficienti vengono calcolati in
        # float64 per ridurre l'errore numerico.
        betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float64,
        )

        alphas = 1.0 - betas

        alphas_cumprod = torch.cumprod(
            alphas,
            dim=0,
        )

        alphas_cumprod_previous = torch.cat(
            [
                torch.ones(
                    1,
                    dtype=torch.float64,
                ),
                alphas_cumprod[:-1],
            ],
            dim=0,
        )

        sqrt_alphas_cumprod = torch.sqrt(
            alphas_cumprod
        )

        sqrt_one_minus_alphas_cumprod = (
            torch.sqrt(
                1.0 - alphas_cumprod
            )
        )

        sqrt_recip_alphas = torch.sqrt(
            1.0 / alphas
        )

        sqrt_recip_alphas_cumprod = (
            torch.sqrt(
                1.0 / alphas_cumprod
            )
        )

        sqrt_recipm1_alphas_cumprod = (
            torch.sqrt(
                1.0
                / alphas_cumprod
                - 1.0
            )
        )

        posterior_variance = (
            betas
            * (
                1.0
                - alphas_cumprod_previous
            )
            / (
                1.0
                - alphas_cumprod
            )
        )

        posterior_log_variance_clipped = (
            torch.log(
                posterior_variance.clamp(
                    min=1e-20
                )
            )
        )

        posterior_mean_coefficient_1 = (
            betas
            * torch.sqrt(
                alphas_cumprod_previous
            )
            / (
                1.0
                - alphas_cumprod
            )
        )

        posterior_mean_coefficient_2 = (
            (
                1.0
                - alphas_cumprod_previous
            )
            * torch.sqrt(alphas)
            / (
                1.0
                - alphas_cumprod
            )
        )

        buffers = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": (
                alphas_cumprod
            ),
            "alphas_cumprod_previous": (
                alphas_cumprod_previous
            ),
            "sqrt_alphas_cumprod": (
                sqrt_alphas_cumprod
            ),
            "sqrt_one_minus_alphas_cumprod": (
                sqrt_one_minus_alphas_cumprod
            ),
            "sqrt_recip_alphas": (
                sqrt_recip_alphas
            ),
            "sqrt_recip_alphas_cumprod": (
                sqrt_recip_alphas_cumprod
            ),
            "sqrt_recipm1_alphas_cumprod": (
                sqrt_recipm1_alphas_cumprod
            ),
            "posterior_variance": (
                posterior_variance
            ),
            "posterior_log_variance_clipped": (
                posterior_log_variance_clipped
            ),
            "posterior_mean_coefficient_1": (
                posterior_mean_coefficient_1
            ),
            "posterior_mean_coefficient_2": (
                posterior_mean_coefficient_2
            ),
        }

        for name, value in buffers.items():
            self.register_buffer(
                name,
                value.to(torch.float32),
                persistent=True,
            )

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "DDPMSchedule":
        diffusion = config["diffusion"]
        beta_schedule = diffusion[
            "beta_schedule"
        ]

        return cls(
            num_timesteps=int(
                diffusion["num_timesteps"]
            ),
            beta_start=float(
                beta_schedule["beta_start"]
            ),
            beta_end=float(
                beta_schedule["beta_end"]
            ),
        )

    def q_sample(
        self,
        x_start: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
        generator: Optional[
            torch.Generator
        ] = None,
    ) -> tuple[Tensor, Tensor]:
        if not x_start.is_floating_point():
            raise TypeError(
                "x_start deve essere floating point."
            )

        if noise is not None and generator is not None:
            raise ValueError(
                "Specificare noise oppure generator, "
                "non entrambi."
            )

        if noise is None:
            if generator is None:
                noise = torch.randn_like(
                    x_start
                )
            else:
                noise = torch.randn(
                    x_start.shape,
                    dtype=x_start.dtype,
                    device=x_start.device,
                    generator=generator,
                )

        if noise.shape != x_start.shape:
            raise ValueError(
                "noise e x_start devono avere "
                "la stessa shape."
            )

        if noise.device != x_start.device:
            raise ValueError(
                "noise e x_start devono essere "
                "sullo stesso device."
            )

        signal_scale = extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            x_start,
        )

        noise_scale = extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_start,
        )

        x_t = (
            signal_scale * x_start
            + noise_scale * noise
        )

        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ) -> Tensor:
        if x_t.shape != noise.shape:
            raise ValueError(
                "x_t e noise devono avere "
                "la stessa shape."
            )

        reciprocal_signal = extract(
            self.sqrt_recip_alphas_cumprod,
            timesteps,
            x_t,
        )

        reciprocal_noise = extract(
            self.sqrt_recipm1_alphas_cumprod,
            timesteps,
            x_t,
        )

        return (
            reciprocal_signal * x_t
            - reciprocal_noise * noise
        )

    def q_posterior(
        self,
        x_start: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x_start.shape != x_t.shape:
            raise ValueError(
                "x_start e x_t devono avere "
                "la stessa shape."
            )

        coefficient_1 = extract(
            self.posterior_mean_coefficient_1,
            timesteps,
            x_t,
        )

        coefficient_2 = extract(
            self.posterior_mean_coefficient_2,
            timesteps,
            x_t,
        )

        posterior_mean = (
            coefficient_1 * x_start
            + coefficient_2 * x_t
        )

        posterior_variance = extract(
            self.posterior_variance,
            timesteps,
            x_t,
        )

        posterior_log_variance = extract(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t,
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance,
        )
