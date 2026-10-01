
import json
from pathlib import Path

import pandas as pd

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

from src.data.build_captions import (
    generate_caption,
    load_text_config,
)


CONFIG_PATH = Path("configs/text_conditioning.yaml")
CAPTIONS_PATH = Path("data/processed/captions/metadata_with_captions.csv")
TOKENIZER_PATH = Path("data/processed/tokenizer/tokenizer.json")
TOKENIZER_SUMMARY_PATH = Path("data/processed/tokenizer/tokenizer_summary.json")


def load_artifacts():
    config = load_text_config(CONFIG_PATH)
    captioned = pd.read_csv(CAPTIONS_PATH)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    with TOKENIZER_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    return config, captioned, tokenizer, summary


def test_caption_count_and_ids():
    _, captioned, _, _ = load_artifacts()

    assert len(captioned) == 10_000
    assert captioned["image_id"].nunique() == 10_000
    assert not captioned["image_id"].duplicated().any()


def test_captions_are_not_missing():
    _, captioned, _, _ = load_artifacts()

    assert not captioned["caption"].isna().any()
    assert not captioned["caption"].str.strip().eq("").any()


def test_caption_generation_is_deterministic():
    config, captioned, _, _ = load_artifacts()

    regenerated = [
        generate_caption(row=row, caption_config=config["caption"])
        for row in captioned.to_dict(orient="records")
    ]

    assert regenerated == captioned["caption"].tolist()


def test_ood_caption_contains_selected_values():
    _, captioned, _, _ = load_artifacts()

    ood = captioned[captioned["split"] == "test_ood"]

    assert len(ood) == 144
    assert ood["caption"].str.contains("face shape variant 3", regex=False).all()
    assert ood["caption"].str.contains("hair color variant 3", regex=False).all()


def test_tokenizer_uses_training_only_vocabulary():
    config, captioned, tokenizer, _ = load_artifacts()

    training = captioned[captioned["split"] == "train"]
    whitespace = Whitespace()
    expected_tokens = set()

    for caption in training["caption"]:
        expected_tokens.update(
            token for token, _ in whitespace.pre_tokenize_str(caption)
        )

    expected_tokens.update(config["tokenizer"]["special_tokens"].values())
    actual_tokens = set(tokenizer.get_vocab())

    assert actual_tokens == expected_tokens


def test_special_tokens_exist():
    config, _, tokenizer, summary = load_artifacts()

    special_tokens = config["tokenizer"]["special_tokens"]

    for name, token in special_tokens.items():
        token_id = tokenizer.token_to_id(token)

        assert token_id is not None
        assert summary["special_token_ids"][name] == token_id


def test_no_unknown_tokens_in_dataset():
    config, captioned, tokenizer, _ = load_artifacts()

    unk_token = config["tokenizer"]["special_tokens"]["unk"]
    unk_id = tokenizer.token_to_id(unk_token)

    for caption in captioned["caption"]:
        encoding = tokenizer.encode(caption)
        assert unk_id not in encoding.ids


def test_encoded_length_is_fixed():
    _, captioned, tokenizer, summary = load_artifacts()

    expected_length = summary["maximum_sequence_length"]
    examples = [
        captioned.iloc[0]["caption"],
        captioned.iloc[-1]["caption"],
        "a cartoon avatar",
    ]

    for text in examples:
        encoding = tokenizer.encode(text)

        assert len(encoding.ids) == expected_length
        assert len(encoding.attention_mask) == expected_length


def test_sos_eos_and_padding_positions():
    config, _, tokenizer, summary = load_artifacts()

    special = config["tokenizer"]["special_tokens"]
    sos_id = tokenizer.token_to_id(special["sos"])
    eos_id = tokenizer.token_to_id(special["eos"])
    pad_id = tokenizer.token_to_id(special["pad"])

    encoding = tokenizer.encode("a cartoon avatar")
    active_length = sum(encoding.attention_mask)

    assert encoding.ids[0] == sos_id
    assert encoding.ids[active_length - 1] == eos_id
    assert all(token_id == pad_id for token_id in encoding.ids[active_length:])
    assert len(encoding.ids) == summary["maximum_sequence_length"]
