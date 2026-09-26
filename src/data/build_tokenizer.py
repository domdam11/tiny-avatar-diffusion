
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from src.data.build_captions import (
    load_text_config,
)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def compute_text_sha256(
    image_ids: list[str],
    captions: list[str],
) -> str:
    digest = hashlib.sha256()

    for image_id, caption in zip(
        image_ids,
        captions,
    ):
        line = (
            f"{image_id}\t{caption}\n"
        )
        digest.update(
            line.encode("utf-8")
        )

    return digest.hexdigest()


def build_tokenizer(
    captioned: pd.DataFrame,
    config: dict,
) -> tuple[Tokenizer, dict]:
    required_columns = {
        "image_id",
        "split",
        "caption",
    }

    missing_columns = (
        required_columns
        - set(captioned.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonne mancanti: "
            f"{sorted(missing_columns)}"
        )

    training = (
        captioned[
            captioned["split"] == "train"
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    if training.empty:
        raise ValueError(
            "Training split vuoto."
        )

    tokenizer_config = config["tokenizer"]
    special = tokenizer_config[
        "special_tokens"
    ]

    ordered_special_tokens = [
        special["unk"],
        special["pad"],
        special["sos"],
        special["eos"],
    ]

    tokenizer = Tokenizer(
        WordLevel(
            unk_token=special["unk"]
        )
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        min_frequency=int(
            tokenizer_config[
                "min_frequency"
            ]
        ),
        special_tokens=(
            ordered_special_tokens
        ),
        show_progress=True,
    )

    training_captions = (
        training["caption"]
        .astype(str)
        .tolist()
    )

    tokenizer.train_from_iterator(
        training_captions,
        trainer=trainer,
        length=len(training_captions),
    )

    special_ids = {
        name: tokenizer.token_to_id(token)
        for name, token in special.items()
    }

    missing_special_tokens = [
        name
        for name, token_id
        in special_ids.items()
        if token_id is None
    ]

    if missing_special_tokens:
        raise RuntimeError(
            "Token speciali mancanti: "
            f"{missing_special_tokens}"
        )

    tokenizer.post_processor = (
        TemplateProcessing(
            single=(
                f"{special['sos']} "
                f"$A "
                f"{special['eos']}"
            ),
            special_tokens=[
                (
                    special["sos"],
                    special_ids["sos"],
                ),
                (
                    special["eos"],
                    special_ids["eos"],
                ),
            ],
        )
    )

    training_lengths = [
        len(
            tokenizer.encode(
                caption
            ).ids
        )
        for caption in training_captions
    ]

    max_length = max(training_lengths)

    tokenizer.enable_truncation(
        max_length=max_length
    )

    tokenizer.enable_padding(
        length=max_length,
        pad_id=special_ids["pad"],
        pad_token=special["pad"],
    )

    unknown_counts = {}

    for split_name, frame in captioned.groupby(
        "split"
    ):
        unknown_counts[split_name] = sum(
            encoding.ids.count(
                special_ids["unk"]
            )
            for encoding in tokenizer.encode_batch(
                frame["caption"]
                .astype(str)
                .tolist()
            )
        )

    summary = {
        "tokenizer_model": "WordLevel",
        "pre_tokenizer": "Whitespace",
        "training_examples": int(
            len(training)
        ),
        "vocabulary_size": int(
            tokenizer.get_vocab_size()
        ),
        "maximum_sequence_length": int(
            max_length
        ),
        "minimum_training_sequence_length": int(
            min(training_lengths)
        ),
        "maximum_training_sequence_length": int(
            max(training_lengths)
        ),
        "special_tokens": special,
        "special_token_ids": special_ids,
        "unknown_token_counts": {
            key: int(value)
            for key, value
            in unknown_counts.items()
        },
        "training_captions_sha256": (
            compute_text_sha256(
                image_ids=training[
                    "image_id"
                ].astype(str).tolist(),
                captions=training_captions,
            )
        ),
    }

    return tokenizer, summary


def save_tokenizer(
    tokenizer: Tokenizer,
    summary: dict,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_path = (
        output_dir / "tokenizer.json"
    )

    tokenizer.save(
        str(tokenizer_path)
    )

    vocabulary = tokenizer.get_vocab()

    vocabulary_by_id = dict(
        sorted(
            vocabulary.items(),
            key=lambda item: item[1],
        )
    )

    vocabulary_path = (
        output_dir / "vocabulary.json"
    )

    with vocabulary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vocabulary_by_id,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary["tokenizer_sha256"] = (
        compute_sha256(tokenizer_path)
    )

    summary["vocabulary_sha256"] = (
        compute_sha256(vocabulary_path)
    )

    summary_path = (
        output_dir
        / "tokenizer_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--captions",
        type=Path,
        default=Path(
            "data/processed/captions/"
            "metadata_with_captions.csv"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/text_conditioning.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/tokenizer"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    captioned = pd.read_csv(
        args.captions
    )

    config = load_text_config(
        args.config
    )

    tokenizer, summary = build_tokenizer(
        captioned=captioned,
        config=config,
    )

    summary = save_tokenizer(
        tokenizer=tokenizer,
        summary=summary,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
