
import pytest
import torch

from src.diffusion.sampler import (
    p_mean_variance,
    p_sample,
    sample_loop,
)
from src.diffusion.schedule import (
    DDPMSchedule,
)


def build_small_schedule():
    return DDPMSchedule(
        num_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
    )


def zero_predictor(
    x_t,
    timesteps,
):
    del timesteps
    return torch.zeros_like(x_t)


def test_p_mean_variance_shapes():
    schedule = build_small_schedule()

    x_t = torch.randn(
        4,
        3,
        8,
        8,
    )

    timesteps = torch.tensor(
        [0, 1, 10, 19],
        dtype=torch.long,
    )

    outputs = p_mean_variance(
        schedule=schedule,
        x_t=x_t,
        timesteps=timesteps,
        predicted_noise=torch.zeros_like(
            x_t
        ),
    )

    for output in outputs:
        assert output.shape[0] == 4
        assert torch.isfinite(
            output
        ).all()


def test_no_noise_at_timestep_zero():
    schedule = build_small_schedule()

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.zeros(
        2,
        dtype=torch.long,
    )

    first_noise = torch.randn_like(x_t)
    second_noise = torch.randn_like(x_t)

    first, _ = p_sample(
        schedule=schedule,
        noise_predictor=zero_predictor,
        x_t=x_t,
        timesteps=timesteps,
        noise=first_noise,
    )

    second, _ = p_sample(
        schedule=schedule,
        noise_predictor=zero_predictor,
        x_t=x_t,
        timesteps=timesteps,
        noise=second_noise,
    )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_reverse_step_adds_noise_above_zero():
    schedule = build_small_schedule()

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.full(
        (2,),
        10,
        dtype=torch.long,
    )

    first, _ = p_sample(
        schedule=schedule,
        noise_predictor=zero_predictor,
        x_t=x_t,
        timesteps=timesteps,
        noise=torch.zeros_like(x_t),
    )

    second, _ = p_sample(
        schedule=schedule,
        noise_predictor=zero_predictor,
        x_t=x_t,
        timesteps=timesteps,
        noise=torch.ones_like(x_t),
    )

    assert not torch.equal(
        first,
        second,
    )


def test_step_is_reproducible():
    schedule = build_small_schedule()

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.full(
        (2,),
        10,
        dtype=torch.long,
    )

    first_generator = (
        torch.Generator()
        .manual_seed(42)
    )

    second_generator = (
        torch.Generator()
        .manual_seed(42)
    )

    first, _ = p_sample(
        schedule=schedule,
        noise_predictor=zero_predictor,
        x_t=x_t,
        timesteps=timesteps,
        generator=first_generator,
    )

    second, _ = p_sample(
        schedule=schedule,
        noise_predictor=zero_predictor,
        x_t=x_t,
        timesteps=timesteps,
        generator=second_generator,
    )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_sampling_loop_shape():
    schedule = build_small_schedule()

    samples, intermediates = sample_loop(
        schedule=schedule,
        noise_predictor=zero_predictor,
        shape=(3, 3, 8, 8),
        device=torch.device("cpu"),
        generator=(
            torch.Generator()
            .manual_seed(123)
        ),
        record_every=5,
    )

    assert samples.shape == (
        3,
        3,
        8,
        8,
    )

    assert torch.isfinite(samples).all()
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
    assert len(intermediates) > 0


def test_sampling_loop_is_reproducible():
    schedule = build_small_schedule()

    first, _ = sample_loop(
        schedule=schedule,
        noise_predictor=zero_predictor,
        shape=(2, 3, 8, 8),
        device=torch.device("cpu"),
        generator=(
            torch.Generator()
            .manual_seed(456)
        ),
    )

    second, _ = sample_loop(
        schedule=schedule,
        noise_predictor=zero_predictor,
        shape=(2, 3, 8, 8),
        device=torch.device("cpu"),
        generator=(
            torch.Generator()
            .manual_seed(456)
        ),
    )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_invalid_prediction_shape():
    schedule = build_small_schedule()

    def invalid_predictor(
        x_t,
        timesteps,
    ):
        del timesteps
        return x_t[:, :1]

    with pytest.raises(ValueError):
        p_sample(
            schedule=schedule,
            noise_predictor=(
                invalid_predictor
            ),
            x_t=torch.randn(
                2,
                3,
                8,
                8,
            ),
            timesteps=torch.tensor(
                [1, 1],
                dtype=torch.long,
            ),
        )
