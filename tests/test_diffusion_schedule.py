
from pathlib import Path

import pytest
import torch

from src.diffusion.schedule import (
    DDPMSchedule,
    extract,
    load_diffusion_config,
)


CONFIG_PATH = Path(
    "configs/diffusion.yaml"
)


def build_schedule():
    config = load_diffusion_config(
        CONFIG_PATH
    )

    return DDPMSchedule.from_config(
        config
    )


def test_configuration():
    config = load_diffusion_config(
        CONFIG_PATH
    )

    assert (
        config["diffusion"][
            "num_timesteps"
        ]
        == 1000
    )

    assert (
        config["diffusion"][
            "prediction_type"
        ]
        == "epsilon"
    )

    assert (
        config["data"]["image_size"]
        == 32
    )

    assert (
        config["data"]["image_channels"]
        == 3
    )


def test_schedule_has_no_trainable_parameters():
    schedule = build_schedule()

    assert sum(
        parameter.numel()
        for parameter
        in schedule.parameters()
    ) == 0


def test_schedule_shapes():
    schedule = build_schedule()

    expected_shape = (1000,)

    assert schedule.betas.shape == expected_shape
    assert schedule.alphas.shape == expected_shape

    assert (
        schedule.alphas_cumprod.shape
        == expected_shape
    )

    assert (
        schedule.posterior_variance.shape
        == expected_shape
    )


def test_beta_schedule_endpoints():
    schedule = build_schedule()

    torch.testing.assert_close(
        schedule.betas[0],
        torch.tensor(0.0001),
        rtol=0.0,
        atol=1e-8,
    )

    torch.testing.assert_close(
        schedule.betas[-1],
        torch.tensor(0.02),
        rtol=0.0,
        atol=1e-8,
    )


def test_beta_schedule_is_increasing():
    schedule = build_schedule()

    assert torch.all(
        schedule.betas[1:]
        > schedule.betas[:-1]
    )


def test_alpha_cumprod_is_decreasing():
    schedule = build_schedule()

    assert torch.all(
        schedule.alphas_cumprod > 0
    )

    assert torch.all(
        schedule.alphas_cumprod <= 1
    )

    assert torch.all(
        schedule.alphas_cumprod[1:]
        < schedule.alphas_cumprod[:-1]
    )

    assert (
        schedule.alphas_cumprod[-1]
        < 1e-4
    )


def test_previous_alpha_cumprod():
    schedule = build_schedule()

    torch.testing.assert_close(
        schedule.alphas_cumprod_previous[0],
        torch.tensor(1.0),
    )

    torch.testing.assert_close(
        schedule.alphas_cumprod_previous[1:],
        schedule.alphas_cumprod[:-1],
    )


def test_q_sample_matches_closed_form():
    schedule = build_schedule()

    x_start = torch.linspace(
        -1.0,
        1.0,
        steps=2 * 3 * 8 * 8,
    ).reshape(2, 3, 8, 8)

    noise = torch.linspace(
        1.0,
        -1.0,
        steps=2 * 3 * 8 * 8,
    ).reshape(2, 3, 8, 8)

    timesteps = torch.tensor(
        [0, 999],
        dtype=torch.long,
    )

    x_t, returned_noise = (
        schedule.q_sample(
            x_start=x_start,
            timesteps=timesteps,
            noise=noise,
        )
    )

    expected = (
        schedule
        .sqrt_alphas_cumprod[
            timesteps
        ]
        .reshape(2, 1, 1, 1)
        * x_start
        + schedule
        .sqrt_one_minus_alphas_cumprod[
            timesteps
        ]
        .reshape(2, 1, 1, 1)
        * noise
    )

    torch.testing.assert_close(
        x_t,
        expected,
    )

    torch.testing.assert_close(
        returned_noise,
        noise,
    )


def test_predict_start_inverts_q_sample():
    schedule = build_schedule()

    generator = torch.Generator().manual_seed(
        42
    )

    x_start = torch.rand(
        4,
        3,
        16,
        16,
        generator=generator,
    ) * 2.0 - 1.0

    noise = torch.randn(
        x_start.shape,
        generator=generator,
    )

    timesteps = torch.tensor(
        [0, 50, 500, 999],
        dtype=torch.long,
    )

    x_t, _ = schedule.q_sample(
        x_start=x_start,
        timesteps=timesteps,
        noise=noise,
    )

    reconstructed = (
        schedule.predict_start_from_noise(
            x_t=x_t,
            timesteps=timesteps,
            noise=noise,
        )
    )

    torch.testing.assert_close(
        reconstructed,
        x_start,
        rtol=1e-3,
        atol=2e-4,
    )


def test_q_sample_preserves_shape_dtype():
    schedule = build_schedule()

    x_start = torch.zeros(
        5,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    timesteps = torch.tensor(
        [0, 100, 300, 700, 999],
        dtype=torch.long,
    )

    x_t, noise = schedule.q_sample(
        x_start=x_start,
        timesteps=timesteps,
    )

    assert x_t.shape == x_start.shape
    assert noise.shape == x_start.shape
    assert x_t.dtype == x_start.dtype
    assert noise.dtype == x_start.dtype

    assert torch.isfinite(x_t).all()
    assert torch.isfinite(noise).all()


def test_q_sample_is_reproducible():
    schedule = build_schedule()

    x_start = torch.zeros(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.tensor(
        [10, 900],
        dtype=torch.long,
    )

    first_generator = (
        torch.Generator()
        .manual_seed(123)
    )

    second_generator = (
        torch.Generator()
        .manual_seed(123)
    )

    first, first_noise = (
        schedule.q_sample(
            x_start=x_start,
            timesteps=timesteps,
            generator=first_generator,
        )
    )

    second, second_noise = (
        schedule.q_sample(
            x_start=x_start,
            timesteps=timesteps,
            generator=second_generator,
        )
    )

    torch.testing.assert_close(
        first,
        second,
    )

    torch.testing.assert_close(
        first_noise,
        second_noise,
    )


def test_q_sample_preserves_gradients():
    schedule = build_schedule()

    x_start = torch.randn(
        2,
        3,
        8,
        8,
        requires_grad=True,
    )

    noise = torch.randn_like(
        x_start
    )

    timesteps = torch.tensor(
        [100, 900],
        dtype=torch.long,
    )

    x_t, _ = schedule.q_sample(
        x_start=x_start,
        timesteps=timesteps,
        noise=noise,
    )

    x_t.square().mean().backward()

    assert x_start.grad is not None
    assert torch.isfinite(
        x_start.grad
    ).all()


def test_posterior_at_first_timestep():
    schedule = build_schedule()

    x_start = torch.randn(
        2,
        3,
        8,
        8,
    )

    x_t = torch.randn_like(
        x_start
    )

    timesteps = torch.zeros(
        2,
        dtype=torch.long,
    )

    mean, variance, log_variance = (
        schedule.q_posterior(
            x_start=x_start,
            x_t=x_t,
            timesteps=timesteps,
        )
    )

    torch.testing.assert_close(
        mean,
        x_start,
        rtol=1e-4,
        atol=1e-5,
    )

    torch.testing.assert_close(
        variance,
        torch.zeros_like(variance),
        atol=1e-8,
        rtol=0.0,
    )

    assert torch.isfinite(
        log_variance
    ).all()


def test_invalid_timestep_dtype():
    schedule = build_schedule()

    x_start = torch.zeros(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.tensor(
        [0.0, 1.0]
    )

    with pytest.raises(TypeError):
        schedule.q_sample(
            x_start=x_start,
            timesteps=timesteps,
        )


def test_invalid_timestep_range():
    schedule = build_schedule()

    x_start = torch.zeros(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.tensor(
        [0, 1000],
        dtype=torch.long,
    )

    with pytest.raises(IndexError):
        schedule.q_sample(
            x_start=x_start,
            timesteps=timesteps,
        )


def test_extract_broadcast_shape():
    values = torch.arange(
        10,
        dtype=torch.float32,
    )

    reference = torch.zeros(
        3,
        4,
        5,
        6,
    )

    timesteps = torch.tensor(
        [1, 4, 8],
        dtype=torch.long,
    )

    extracted = extract(
        values=values,
        timesteps=timesteps,
        reference=reference,
    )

    assert extracted.shape == (
        3,
        1,
        1,
        1,
    )

    torch.testing.assert_close(
        extracted.flatten(),
        torch.tensor(
            [1.0, 4.0, 8.0]
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA non disponibile",
)
def test_schedule_on_cuda():
    schedule = (
        build_schedule()
        .to("cuda")
    )

    x_start = torch.randn(
        2,
        3,
        8,
        8,
        device="cuda",
    )

    timesteps = torch.tensor(
        [0, 999],
        dtype=torch.long,
        device="cuda",
    )

    x_t, noise = schedule.q_sample(
        x_start=x_start,
        timesteps=timesteps,
    )

    assert x_t.is_cuda
    assert noise.is_cuda
    assert torch.isfinite(x_t).all()
