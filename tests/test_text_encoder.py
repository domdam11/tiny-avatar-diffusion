from pathlib import Path

import pytest
import torch

from src.models.text_encoder import (
    TextTransformerEncoder,
    load_text_encoder_config,
)

CONFIG_PATH = Path("configs/text_encoder.yaml")


def test_load_text_encoder_config() -> None:
    config = load_text_encoder_config(CONFIG_PATH)

    assert "text_encoder" in config
    assert int(config["text_encoder"]["vocab_size"]) > 0
    assert int(config["text_encoder"]["max_sequence_length"]) > 0
    assert int(config["text_encoder"]["pad_token_id"]) >= 0


def test_text_encoder_output_shape_and_padding() -> None:
    config = load_text_encoder_config(CONFIG_PATH)
    model = TextTransformerEncoder.from_config(config)

    sequence_length = int(config["text_encoder"]["max_sequence_length"])
    d_model = int(config["text_encoder"]["d_model"])
    pad_token_id = int(config["text_encoder"]["pad_token_id"])

    input_ids = torch.full((2, sequence_length), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((2, sequence_length), dtype=torch.long)

    input_ids[0, :5] = torch.tensor([2, 9, 12, 11, 3], dtype=torch.long)
    attention_mask[0, :5] = 1

    input_ids[1, :4] = torch.tensor([2, 14, 7, 3], dtype=torch.long)
    attention_mask[1, :4] = 1

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.shape == (2, sequence_length, d_model)
    assert torch.isfinite(output).all()

    padded_positions = attention_mask == 0
    assert torch.allclose(
        output[padded_positions],
        torch.zeros_like(output[padded_positions]),
        atol=1e-6,
    )


def test_text_encoder_rejects_wrong_dtype() -> None:
    config = load_text_encoder_config(CONFIG_PATH)
    model = TextTransformerEncoder.from_config(config)

    sequence_length = int(config["text_encoder"]["max_sequence_length"])

    # input_ids deve essere torch.long.
    with pytest.raises(TypeError, match="input_ids deve avere dtype torch.long."):
        model(
            input_ids=torch.zeros((1, sequence_length), dtype=torch.float32),
            attention_mask=torch.ones((1, sequence_length), dtype=torch.long),
        )
