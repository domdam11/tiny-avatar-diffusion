from pathlib import Path

import pytest
import torch

from src.diffusion.reverse_sampler import (
    ReverseSampler,
    load_sampling_config,
)


class DummyTextEncoder(torch.nn.Module):
    # Text encoder minimo per testare solo il sampler.

    pad_token_id = 1

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        batch_size, sequence_length = input_ids.shape

        return torch.zeros(
            batch_size,
            sequence_length,
            8,
            device=input_ids.device,
        )


class DummyUNet(torch.nn.Module):
    # Restituisce rumore nullo: il test verifica l'orchestrazione,
    # non la qualità del modello.

    def __init__(self):
        super().__init__()
        self.dummy_parameter = torch.nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        noisyimages,
        timesteps,
        context,
        attention_mask,
    ):
        return torch.zeros_like(noisyimages)


class DummySchedule(torch.nn.Module):
    # Espone lo stesso contratto minimo richiesto dal sampler reale.

    def __init__(self):
        super().__init__()
        self.num_timesteps = 4

    def reverse_step(
        self,
        noisyimages,
        predicted_noise,
        timesteps,
        generator=None,
        clip_denoised=True,
    ):
        # Con rumore predetto nullo, il test può limitarsi a restituire
        # la stessa immagine e una stima grezza di x_0.
        predicted_x_start = noisyimages.clamp(-1.0, 1.0)

        return noisyimages, predicted_x_start


@pytest.fixture()
def sampling_config_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "sampling.yaml"

    config_path.write_text(
        """
sampling:
  num_inference_steps: 1000
  guidance_scale: 1.0
  clip_denoised: true
""".strip(),
        encoding="utf-8",
    )

    return config_path


def test_load_sampling_config_ok(
    sampling_config_file: Path,
) -> None:
    config = load_sampling_config(
        sampling_config_file
    )

    assert config["sampling"]["guidance_scale"] == 1.0
    assert config["sampling"]["clip_denoised"] is True


def test_load_sampling_config_requires_sampling_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "foo: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Sezione 'sampling' mancante.",
    ):
        load_sampling_config(config_path)


def test_sampler_generates_tensor_with_expected_shape() -> None:
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()
    schedule = DummySchedule()

    sampler = ReverseSampler(
        conditional_unet=unet,
        text_encoder=text_encoder,
        schedule=schedule,
        guidance_scale=1.0,
        clip_denoised=True,
        device=torch.device("cpu"),
    )

    input_ids = torch.tensor(
        [
            [2, 3, 4, 1],
            [5, 6, 1, 1],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    generator = torch.Generator().manual_seed(7)

    samples = sampler.sample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_shape=(3, 32, 32),
        generator=generator,
    )

    assert samples.shape == (2, 3, 32, 32)
    assert samples.dtype == torch.float32
    assert torch.all(samples <= 1.0)
    assert torch.all(samples >= -1.0)


def test_sampler_can_return_intermediates() -> None:
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()
    schedule = DummySchedule()

    sampler = ReverseSampler(
        conditional_unet=unet,
        text_encoder=text_encoder,
        schedule=schedule,
        guidance_scale=1.0,
        clip_denoised=True,
        device=torch.device("cpu"),
    )

    input_ids = torch.tensor(
        [[2, 3, 4, 1]],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [[1, 1, 1, 0]],
        dtype=torch.long,
    )

    generator = torch.Generator().manual_seed(11)

    final_sample, intermediates = sampler.sample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_shape=(3, 32, 32),
        generator=generator,
        return_intermediates=True,
        record_every=1,
    )

    assert final_sample.shape == (1, 3, 32, 32)
    assert len(intermediates) == schedule.num_timesteps

    timestep_0, image_0 = intermediates[0]
    assert isinstance(timestep_0, int)
    assert image_0.shape == (1, 3, 32, 32)


def test_sampler_validates_input_dtypes() -> None:
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()
    schedule = DummySchedule()

    sampler = ReverseSampler(
        conditional_unet=unet,
        text_encoder=text_encoder,
        schedule=schedule,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        TypeError,
        match="input_ids deve avere dtype torch.long.",
    ):
        sampler.sample(
            input_ids=torch.zeros(1, 4),
            attention_mask=torch.ones(
                1,
                4,
                dtype=torch.long,
            ),
            image_shape=(3, 32, 32),
        )


def test_sampler_rejects_negative_guidance_scale() -> None:
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()
    schedule = DummySchedule()

    with pytest.raises(
        ValueError,
        match="guidance_scale non può essere negativo.",
    ):
        ReverseSampler(
            conditional_unet=unet,
            text_encoder=text_encoder,
            schedule=schedule,
            guidance_scale=-1.0,
        )
