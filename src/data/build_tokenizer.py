
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer


def compute_sha256(path: Path) -> str:
    # Calcola lo SHA-256 di un file binario
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_text_config(config_path: Path) -> dict:
    # Carica e valida la configurazione per caption e tokenizer
    with Path(config_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    required_sections = {"caption", "tokenizer"}

    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(
            "Sezioni mancanti nella configurazione: "
            f"{sorted(missing_sections)}"
        )

    tokenizer_config = config["tokenizer"]

    required_tokenizer_keys = {
        "model",
        "pre_tokenizer",
        "min_frequency",
        "special_tokens",
    }

    missing_tokenizer_keys = (
        required_tokenizer_keys
        - set(tokenizer_config)
    )

    if missing_tokenizer_keys:
        raise ValueError(
            "Parametri tokenizer mancanti: "
            f"{sorted(missing_tokenizer_keys)}"
        )

    if tokenizer_config["model"] != "word_level":
        raise ValueError(
            "È supportato solo tokenizer.model='word_level'."
        )

    if tokenizer_config["pre_tokenizer"] != "whitespace":
        raise ValueError(
            "È supportato solo "
            "tokenizer.pre_tokenizer='whitespace'."
        )

    if int(tokenizer_config["min_frequency"]) < 1:
        raise ValueError(
            "tokenizer.min_frequency deve essere almeno 1."
        )

    special_tokens = tokenizer_config["special_tokens"]

    required_special_tokens = {
        "unk",
        "pad",
        "sos",
        "eos",
    }

    missing_special_tokens = (
        required_special_tokens - set(special_tokens)
    )

    if missing_special_tokens:
        raise ValueError(
            "Token speciali mancanti: "
            f"{sorted(missing_special_tokens)}"
        )

    token_values = list(special_tokens.values())

    if len(token_values) != len(set(token_values)):
        raise ValueError(
            "I token speciali devono essere tutti distinti."
        )

    return config


def build_tokenizer(
    train_captions: list[str],
    config: dict,
) -> tuple[Tokenizer, dict]:
    # Addestra il tokenizer esclusivamente sulle caption train,
    # poi applica SOS, EOS, truncation e padding.
    if not train_captions:
        raise ValueError(
            "Nessuna caption di training disponibile."
        )

    tokenizer_config = config["tokenizer"]
    special_tokens = tokenizer_config["special_tokens"]

    ordered_special_tokens = [
        special_tokens["unk"],
        special_tokens["pad"],
        special_tokens["sos"],
        special_tokens["eos"],
    ]

    tokenizer = Tokenizer(
        WordLevel(
            unk_token=special_tokens["unk"]
        )
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        min_frequency=int(
            tokenizer_config["min_frequency"]
        ),
        special_tokens=ordered_special_tokens,
        show_progress=True,
    )

    # Il vocabolario viene costruito solo dal training split:
    # validation, test ordinary e OOD non contribuiscono.
    tokenizer.train_from_iterator(
        train_captions,
        trainer=trainer,
        length=len(train_captions),
    )

    special_token_ids = {
        name: tokenizer.token_to_id(token)
        for name, token in special_tokens.items()
    }

    missing_ids = [
        name
        for name, token_id in special_token_ids.items()
        if token_id is None
    ]

    if missing_ids:
        raise RuntimeError(
            "Token speciali senza ID: "
            f"{missing_ids}"
        )

    # Calcoliamo la massima lunghezza della sola caption,
    # prima di aggiungere i delimitatori SOS ed EOS.
    raw_lengths = [
        len(
            tokenizer.encode(
                caption,
                add_special_tokens=False,
            ).ids
        )
        for caption in train_captions
    ]

    max_train_length_without_boundaries = max(
        raw_lengths
    )

    # Ogni sequenza avrà: [SOS] + caption + [EOS].
    maximum_sequence_length = (
        max_train_length_without_boundaries + 2
    )

    # Aggiunge automaticamente SOS e EOS a ogni caption.
    tokenizer.post_processor = TemplateProcessing(
        single=(
            f"{special_tokens['sos']} "
            f"$A "
            f"{special_tokens['eos']}"
        ),
        special_tokens=[
            (
                special_tokens["sos"],
                special_token_ids["sos"],
            ),
            (
                special_tokens["eos"],
                special_token_ids["eos"],
            ),
        ],
    )

    # Impone una lunghezza fissa per i batch del Transformer.
    tokenizer.enable_truncation(
        max_length=maximum_sequence_length
    )

    tokenizer.enable_padding(
        length=maximum_sequence_length,
        pad_id=special_token_ids["pad"],
        pad_token=special_tokens["pad"],
    )

    return tokenizer, {
        "max_train_length_without_boundaries": (
            max_train_length_without_boundaries
        ),
        "maximum_sequence_length": (
            maximum_sequence_length
        ),
        "special_token_ids": special_token_ids,
    }


def build_summary(
    tokenizer: Tokenizer,
    captioned: pd.DataFrame,
    train_captions: list[str],
    special_tokens: dict,
    tokenizer_info: dict,
) -> dict:
    # Crea un summary verificabile degli artefatti tokenizer
    encoded_train = tokenizer.encode_batch(
        train_captions
    )

    encoded_lengths = {
        len(encoding.ids)
        for encoding in encoded_train
    }

    if encoded_lengths != {
        tokenizer_info["maximum_sequence_length"]
    }:
        raise RuntimeError(
            "Le sequenze train non hanno lunghezza fissa. "
            f"Osservate: {sorted(encoded_lengths)}"
        )

    unknown_token_id = tokenizer_info[
        "special_token_ids"
    ]["unk"]

    unknown_token_counts = {}

    for split_name, split_frame in captioned.groupby(
        "split"
    ):
        encodings = tokenizer.encode_batch(
            split_frame["caption"]
            .astype(str)
            .tolist()
        )

        unknown_token_counts[split_name] = sum(
            encoding.ids.count(unknown_token_id)
            for encoding in encodings
        )

    return {
        "tokenizer_model": "WordLevel",
        "pre_tokenizer": "Whitespace",
        "num_train_captions": int(
            len(train_captions)
        ),
        "training_examples": int(
            len(train_captions)
        ),
        "vocabulary_size": int(
            tokenizer.get_vocab_size()
        ),
        "special_tokens": special_tokens,
        "special_token_ids": {
            name: int(token_id)
            for name, token_id in tokenizer_info[
                "special_token_ids"
            ].items()
        },
        "max_train_length_without_boundaries": int(
            tokenizer_info[
                "max_train_length_without_boundaries"
            ]
        ),
        "maximum_sequence_length": int(
            tokenizer_info["maximum_sequence_length"]
        ),
        # Chiave mantenuta per compatibilità con i test
        # e il DataLoader già costruiti nel progetto.
        "sequence_length_with_boundaries": int(
            tokenizer_info["maximum_sequence_length"]
        ),
        "unknown_token_counts": {
            split_name: int(count)
            for split_name, count
            in unknown_token_counts.items()
        },
    }


def main() -> None:
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

    args = parser.parse_args()

    if not args.captions.is_file():
        raise FileNotFoundError(
            f"File caption non trovato: {args.captions}"
        )

    captioned = pd.read_csv(args.captions)

    required_columns = {
        "image_id",
        "split",
        "caption",
    }

    missing_columns = (
        required_columns - set(captioned.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel file caption: "
            f"{sorted(missing_columns)}"
        )

    config = load_text_config(args.config)

    train_captions = (
        captioned.loc[
            captioned["split"] == "train",
            "caption",
        ]
        .astype(str)
        .tolist()
    )

    tokenizer, tokenizer_info = build_tokenizer(
        train_captions=train_captions,
        config=config,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_path = (
        args.output_dir / "tokenizer.json"
    )

    # Il salvataggio avviene soltanto dopo aver configurato
    # post-processing, truncation e padding.
    tokenizer.save(str(tokenizer_path))

    special_tokens = config["tokenizer"][
        "special_tokens"
    ]

    summary = build_summary(
        tokenizer=tokenizer,
        captioned=captioned,
        train_captions=train_captions,
        special_tokens=special_tokens,
        tokenizer_info=tokenizer_info,
    )

    summary["tokenizer_sha256"] = compute_sha256(
        tokenizer_path
    )

    summary_path = (
        args.output_dir / "tokenizer_summary.json"
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

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
