
from pathlib import Path

import pytest
import torch

from src.models.conditional_unet import (
    ConditionalUNet,
    CrossAttention2D,
    SinusoidalTimeEmbedding,
    load_unet_config,
)


CONFIG_PATH = Path(
    "configs/unet.yaml"
)


@pytest.fixture(scope="module")
def config():
    return load_unet_config(
        CONFIG_PATH
    )


@pytest.fixture(scope="module")
def conditional_model(config):
    return ConditionalUNet.from_config(
        config,
        conditional=True,
    ).eval()


@pytest.fixture(scope="module")
def unconditional_model(config):
    return ConditionalUNet.from_config(
        config,
        conditional=False,
    ).eval()


def test_configuration(config):
    model_config = config["unet"]

    assert model_config[
        "image_channels"
    ] == 3

    assert model_config[
        "base_channels"
    ] == 64

    assert model_config[
        "channel_multipliers"
    ] == [1, 2, 4]

    assert model_config[
        "context_dim"
    ] == 128


def test_time_embedding():
    module = SinusoidalTimeEmbedding(
        embedding_dim=64
    )

    timesteps = torch.tensor(
        [0, 1, 500, 999],
        dtype=torch.long,
    )

    embedding = module(timesteps)

    assert embedding.shape == (4, 64)
    assert torch.isfinite(embedding).all()

    second = module(timesteps)

    torch.testing.assert_close(
        embedding,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_cross_attention_shape():
    attention = CrossAttention2D(
        channels=64,
        context_dim=128,
        num_heads=4,
        normalization_groups=8,
        dropout=0.0,
    ).eval()

    hidden_states = torch.randn(
        2,
        64,
        8,
        8,
    )

    context = torch.randn(
        2,
        44,
        128,
    )

    mask = torch.ones(
        2,
        44,
        dtype=torch.long,
    )

    output = attention(
        hidden_states,
        context,
        mask,
    )

    assert output.shape == (
        2,
        64,
        8,
        8,
    )

    assert torch.isfinite(output).all()


def test_masked_tokens_do_not_affect_attention():
    torch.manual_seed(42)

    attention = CrossAttention2D(
        channels=64,
        context_dim=128,
        num_heads=4,
        normalization_groups=8,
        dropout=0.0,
    ).eval()

    hidden_states = torch.randn(
        1,
        64,
        4,
        4,
    )

    first_context = torch.randn(
        1,
        6,
        128,
    )

    second_context = first_context.clone()

    second_context[:, 4:] = (
        torch.randn_like(
            second_context[:, 4:]
        )
        * 1000.0
    )

    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0]],
        dtype=torch.long,
    )

    first_output = attention(
        hidden_states,
        first_context,
        mask,
    )

    second_output = attention(
        hidden_states,
        second_context,
        mask,
    )

    torch.testing.assert_close(
        first_output,
        second_output,
        rtol=1e-5,
        atol=1e-5,
    )


def test_conditional_output_shape(
    conditional_model,
):
    noisy_images = torch.randn(
        2,
        3,
        32,
        32,
    )

    timesteps = torch.tensor(
        [0, 999],
        dtype=torch.long,
    )

    context = torch.randn(
        2,
        44,
        128,
    )

    mask = torch.ones(
        2,
        44,
        dtype=torch.long,
    )

    with torch.no_grad():
        output = conditional_model(
            noisy_images=noisy_images,
            timesteps=timesteps,
            context=context,
            attention_mask=mask,
        )

    assert output.shape == noisy_images.shape
    assert torch.isfinite(output).all()


def test_unconditional_output_shape(
    unconditional_model,
):
    noisy_images = torch.randn(
        2,
        3,
        32,
        32,
    )

    timesteps = torch.tensor(
        [10, 900],
        dtype=torch.long,
    )

    with torch.no_grad():
        output = unconditional_model(
            noisy_images=noisy_images,
            timesteps=timesteps,
        )

    assert output.shape == noisy_images.shape
    assert torch.isfinite(output).all()


def test_conditioning_is_required(
    conditional_model,
):
    noisy_images = torch.randn(
        1,
        3,
        32,
        32,
    )

    timesteps = torch.tensor(
        [100],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        conditional_model(
            noisy_images=noisy_images,
            timesteps=timesteps,
        )


def test_unconditional_rejects_context(
    unconditional_model,
):
    noisy_images = torch.randn(
        1,
        3,
        32,
        32,
    )

    timesteps = torch.tensor(
        [100],
        dtype=torch.long,
    )

    context = torch.randn(
        1,
        44,
        128,
    )

    mask = torch.ones(
        1,
        44,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        unconditional_model(
            noisy_images=noisy_images,
            timesteps=timesteps,
            context=context,
            attention_mask=mask,
        )


def test_backward_reaches_context(
    conditional_model,
):
    model = conditional_model.train()

    noisy_images = torch.randn(
        1,
        3,
        32,
        32,
    )

    timesteps = torch.tensor(
        [500],
        dtype=torch.long,
    )

    context = torch.randn(
        1,
        44,
        128,
        requires_grad=True,
    )

    mask = torch.ones(
        1,
        44,
        dtype=torch.long,
    )

    output = model(
        noisy_images=noisy_images,
        timesteps=timesteps,
        context=context,
        attention_mask=mask,
    )

    loss = output.square().mean()
    loss.backward()

    assert context.grad is not None
    assert torch.isfinite(
        context.grad
    ).all()

    assert context.grad.norm() > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA non disponibile",
)
def test_conditional_model_cuda(config):
    model = (
        ConditionalUNet.from_config(
            config,
            conditional=True,
        )
        .to("cuda")
        .eval()
    )

    noisy_images = torch.randn(
        2,
        3,
        32,
        32,
        device="cuda",
    )

    timesteps = torch.tensor(
        [0, 999],
        dtype=torch.long,
        device="cuda",
    )

    context = torch.randn(
        2,
        44,
        128,
        device="cuda",
    )

    mask = torch.ones(
        2,
        44,
        dtype=torch.long,
        device="cuda",
    )

    with torch.no_grad():
        output = model(
            noisy_images=noisy_images,
            timesteps=timesteps,
            context=context,
            attention_mask=mask,
        )

    assert output.is_cuda
    assert output.shape == noisy_images.shape
    assert torch.isfinite(output).all()
