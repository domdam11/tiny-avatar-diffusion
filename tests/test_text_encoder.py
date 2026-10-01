
from pathlib import Path

import pytest
import torch

from tokenizers import Tokenizer

from src.data.avatar_dataset import (
    build_dataloader,
    build_datasets,
    load_data_config,
)
from src.models.text_encoder import (
    TextTransformerEncoder,
    load_text_encoder_config,
)


TEXT_CONFIG_PATH = Path(
    "configs/text_encoder.yaml"
)

DATA_CONFIG_PATH = Path(
    "configs/data.yaml"
)


@pytest.fixture(scope="module")
def text_config():
    return load_text_encoder_config(
        TEXT_CONFIG_PATH
    )


@pytest.fixture(scope="module")
def model(text_config):
    return TextTransformerEncoder.from_config(
        text_config
    )


@pytest.fixture(scope="module")
def real_batch():
    data_config = load_data_config(
        DATA_CONFIG_PATH
    )

    datasets = build_datasets(
        data_config
    )

    loader = build_dataloader(
        dataset=datasets["train"],
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        seed=2026,
    )

    return next(iter(loader))


def test_config_matches_tokenizer(
    text_config,
):
    tokenizer = Tokenizer.from_file(
        "data/processed/tokenizer/tokenizer.json"
    )

    encoder_config = text_config[
        "text_encoder"
    ]

    assert tokenizer.get_vocab_size() == (
        encoder_config["vocab_size"]
    )

    assert (
        encoder_config[
            "max_sequence_length"
        ]
        == 39
    )

    assert (
        encoder_config["pad_token_id"]
        == 1
    )


def test_parameter_count_is_compact(
    model,
):
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert 500_000 < parameter_count < 2_000_000


def test_output_shape_and_values(
    model,
    real_batch,
):
    model.eval()

    with torch.no_grad():
        output = model(
            input_ids=real_batch[
                "input_ids"
            ],
            attention_mask=real_batch[
                "attention_mask"
            ],
        )

    assert output.shape == (
        8,
        39,
        128,
    )

    assert torch.isfinite(output).all()


def test_eval_is_deterministic(
    model,
    real_batch,
):
    model.eval()

    with torch.no_grad():
        first = model(
            real_batch["input_ids"],
            real_batch["attention_mask"],
        )

        second = model(
            real_batch["input_ids"],
            real_batch["attention_mask"],
        )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_padding_outputs_are_zero(
    model,
):
    model.eval()

    input_ids = torch.tensor(
        [
            [2, 9, 12, 3, 1, 1],
            [2, 9, 12, 11, 20, 3],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask,
        )

    torch.testing.assert_close(
        output[0, 4:],
        torch.zeros_like(output[0, 4:]),
        rtol=0.0,
        atol=0.0,
    )


def test_masked_tokens_do_not_affect_output(
    model,
):
    model.eval()

    first_ids = torch.tensor(
        [[2, 9, 12, 3, 1, 1]],
        dtype=torch.long,
    )

    second_ids = torch.tensor(
        [[2, 9, 12, 3, 40, 70]],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0]],
        dtype=torch.long,
    )

    with torch.no_grad():
        first_output = model(
            first_ids,
            attention_mask,
        )

        second_output = model(
            second_ids,
            attention_mask,
        )

    torch.testing.assert_close(
        first_output[:, :4],
        second_output[:, :4],
        rtol=1e-6,
        atol=1e-6,
    )


def test_content_changes_representation(
    model,
):
    model.eval()

    first_ids = torch.tensor(
        [[2, 9, 12, 11, 20, 3]],
        dtype=torch.long,
    )

    second_ids = torch.tensor(
        [[2, 9, 12, 11, 21, 3]],
        dtype=torch.long,
    )

    mask = torch.ones_like(first_ids)

    with torch.no_grad():
        first_output = model(
            first_ids,
            mask,
        )

        second_output = model(
            second_ids,
            mask,
        )

    assert not torch.allclose(
        first_output,
        second_output,
    )


def test_masked_mean_pool(
    model,
    real_batch,
):
    model.eval()

    with torch.no_grad():
        output = model(
            real_batch["input_ids"],
            real_batch["attention_mask"],
        )

        pooled = model.masked_mean_pool(
            output,
            real_batch[
                "attention_mask"
            ],
        )

    assert pooled.shape == (
        8,
        128,
    )

    assert torch.isfinite(pooled).all()


def test_gradients_are_finite(
    text_config,
    real_batch,
):
    model = (
        TextTransformerEncoder.from_config(
            text_config
        )
    )

    model.train()

    output = model(
        real_batch["input_ids"],
        real_batch["attention_mask"],
    )

    loss = output.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    gradient_norm = torch.sqrt(
        sum(
            gradient.float().square().sum()
            for gradient in gradients
        )
    )

    assert gradient_norm > 0


def test_state_dict_round_trip(
    text_config,
    real_batch,
    tmp_path,
):
    first_model = (
        TextTransformerEncoder.from_config(
            text_config
        )
    )

    first_model.eval()

    checkpoint_path = (
        tmp_path / "text_encoder.pt"
    )

    torch.save(
        first_model.state_dict(),
        checkpoint_path,
    )

    second_model = (
        TextTransformerEncoder.from_config(
            text_config
        )
    )

    second_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    second_model.eval()

    with torch.no_grad():
        first_output = first_model(
            real_batch["input_ids"],
            real_batch["attention_mask"],
        )

        second_output = second_model(
            real_batch["input_ids"],
            real_batch["attention_mask"],
        )

    torch.testing.assert_close(
        first_output,
        second_output,
        rtol=0.0,
        atol=0.0,
    )


def test_invalid_token_id(
    model,
):
    input_ids = torch.tensor(
        [[2, 9, 132, 3]],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    with pytest.raises(IndexError):
        model(
            input_ids,
            attention_mask,
        )


def test_empty_sequence_is_rejected(
    model,
):
    input_ids = torch.ones(
        2,
        5,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        2,
        5,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(
            input_ids,
            attention_mask,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA non disponibile",
)
def test_cuda_forward(
    text_config,
    real_batch,
):
    model = (
        TextTransformerEncoder.from_config(
            text_config
        )
        .to("cuda")
        .eval()
    )

    input_ids = real_batch[
        "input_ids"
    ].to("cuda")

    attention_mask = real_batch[
        "attention_mask"
    ].to("cuda")

    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask,
        )

    assert output.shape == (
        8,
        39,
        128,
    )

    assert output.is_cuda
    assert torch.isfinite(output).all()
