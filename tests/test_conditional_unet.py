
from pathlib import Path

import pytest
import torch

from src.models.conditional_unet import (
    ConditionalUNet,
    load_unet_config,
)


CONFIG_PATH = Path("configs/unet.yaml")


@pytest.fixture(scope="module")
def config() -> dict:
    return load_unet_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def conditional_unet(config: dict) -> ConditionalUNet:
    torch.manual_seed(2026)

    model = ConditionalUNet.from_config(
        config,
        conditional=True,
    )

    model.eval()

    return model


@pytest.fixture(scope="module")
def unconditional_unet(config: dict) -> ConditionalUNet:
    torch.manual_seed(2026)

    model = ConditionalUNet.from_config(
        config,
        conditional=False,
    )

    model.eval()

    return model


def build_conditional_inputs(
    batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Costruisce un batch sintetico compatibile con le configurazioni
    # ufficiali del progetto.

    torch.manual_seed(2026)

    noisyimages = torch.randn(
        batch_size,
        3,
        32,
        32,
    )

    timesteps = torch.tensor(
        [10, 750][:batch_size],
        dtype=torch.long,
    )

    context = torch.randn(
        batch_size,
        44,
        128,
    )

    attention_mask = torch.zeros(
        batch_size,
        44,
        dtype=torch.long,
    )

    attention_mask[:, :8] = 1

    return (
        noisyimages,
        timesteps,
        context,
        attention_mask,
    )


def test_load_unet_config(config: dict) -> None:
    unet_config = config["unet"]

    assert unet_config["image_channels"] == 3
    assert unet_config["output_channels"] == 3
    assert unet_config["base_channels"] == 64

    assert unet_config["channel_multipliers"] == [
        1,
        2,
        4,
    ]

    assert unet_config["num_res_blocks"] == 2
    assert unet_config["time_embedding_dim"] == 256
    assert unet_config["context_dim"] == 128
    assert unet_config["attention_levels"] == [1, 2]
    assert unet_config["attention_heads"] == 4


def test_conditional_unet_is_a_real_multiresolution_unet(
    conditional_unet: ConditionalUNet,
) -> None:
    # Verifica strutturale: la configurazione deve governare davvero
    # encoder, decoder, downsampling, upsampling e skip connection.

    assert len(conditional_unet.channels) == 3
    assert conditional_unet.channels == [64, 128, 256]

    assert len(conditional_unet.down_blocks) == 3
    assert len(conditional_unet.up_blocks) == 3

    # I primi due livelli encoder riducono la risoluzione.
    assert conditional_unet.down_blocks[0].downsample is not None
    assert conditional_unet.down_blocks[1].downsample is not None
    assert conditional_unet.down_blocks[2].downsample is None

    # I primi due blocchi decoder, ordinati dal livello profondo
    # verso quello superficiale, riportano alla risoluzione iniziale.
    assert conditional_unet.up_blocks[0].upsample is not None
    assert conditional_unet.up_blocks[1].upsample is not None
    assert conditional_unet.up_blocks[2].upsample is None

    # I blocchi residuali configurati sono realmente presenti.
    for down_block in conditional_unet.down_blocks:
        assert len(down_block.residual_blocks) == 2

    for up_block in conditional_unet.up_blocks:
        assert len(up_block.residual_blocks) == 2

    # attention_levels=[1, 2]: nessuna attention al livello 0,
    # attention ai due livelli più profondi.
    assert conditional_unet.down_blocks[0].attention is None
    assert conditional_unet.down_blocks[1].attention is not None
    assert conditional_unet.down_blocks[2].attention is not None

    assert conditional_unet.middle_block.attention is not None

    # L'ordine del decoder è: livello 2, livello 1, livello 0.
    assert conditional_unet.up_blocks[0].attention is not None
    assert conditional_unet.up_blocks[1].attention is not None
    assert conditional_unet.up_blocks[2].attention is None


def test_conditional_forward_preserves_image_shape(
    conditional_unet: ConditionalUNet,
) -> None:
    (
        noisyimages,
        timesteps,
        context,
        attention_mask,
    ) = build_conditional_inputs()

    with torch.inference_mode():
        predicted_noise = conditional_unet(
            noisyimages=noisyimages,
            timesteps=timesteps,
            context=context,
            attention_mask=attention_mask,
        )

    assert predicted_noise.shape == noisyimages.shape
    assert torch.isfinite(predicted_noise).all()


def test_unconditional_forward_preserves_image_shape(
    unconditional_unet: ConditionalUNet,
) -> None:
    torch.manual_seed(2026)

    noisyimages = torch.randn(2, 3, 32, 32)

    timesteps = torch.tensor(
        [10, 750],
        dtype=torch.long,
    )

    with torch.inference_mode():
        predicted_noise = unconditional_unet(
            noisyimages=noisyimages,
            timesteps=timesteps,
            context=None,
            attention_mask=None,
        )

    assert predicted_noise.shape == noisyimages.shape
    assert torch.isfinite(predicted_noise).all()


def test_conditional_model_requires_text_context(
    conditional_unet: ConditionalUNet,
) -> None:
    noisyimages = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([10], dtype=torch.long)

    with pytest.raises(ValueError, match="Context"):
        conditional_unet(
            noisyimages=noisyimages,
            timesteps=timesteps,
            context=None,
            attention_mask=None,
        )


def test_unet_has_meaningful_capacity(
    conditional_unet: ConditionalUNet,
    unconditional_unet: ConditionalUNet,
) -> None:
    # Evita una nuova regressione alla rete piatta da circa 0.3 M
    # parametri: il denoiser deve avere capacità adeguata al dataset.

    conditional_parameters = sum(
        parameter.numel()
        for parameter in conditional_unet.parameters()
    )

    unconditional_parameters = sum(
        parameter.numel()
        for parameter in unconditional_unet.parameters()
    )

    assert conditional_parameters > 5_000_000
    assert unconditional_parameters > 5_000_000

    # La cross-attention rende la versione conditional più grande.
    assert conditional_parameters > unconditional_parameters
