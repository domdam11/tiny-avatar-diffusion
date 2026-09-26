
import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from src.data.build_metadata import EXPECTED_VARIANTS
from src.data.build_splits import VALID_SPLITS


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_text_config(config_path: Path) -> dict:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "caption" not in config:
        raise ValueError(
            "Sezione 'caption' mancante."
        )

    if "tokenizer" not in config:
        raise ValueError(
            "Sezione 'tokenizer' mancante."
        )

    caption_config = config["caption"]

    if not caption_config.get("prefix"):
        raise ValueError(
            "Prefix della caption mancante."
        )

    attributes = caption_config.get(
        "attributes",
        [],
    )

    if not attributes:
        raise ValueError(
            "Nessun attributo configurato."
        )

    keys = [
        item["key"]
        for item in attributes
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Attributi duplicati nella configurazione."
        )

    unknown_attributes = sorted(
        set(keys) - set(EXPECTED_VARIANTS)
    )

    if unknown_attributes:
        raise ValueError(
            "Attributi sconosciuti: "
            f"{unknown_attributes}"
        )

    return config


def generate_caption(
    row: Mapping,
    caption_config: dict,
) -> str:
    prefix = caption_config["prefix"].strip()
    fragments = []

    for item in caption_config["attributes"]:
        attribute = item["key"]
        phrase = item["phrase"].strip()

        if attribute not in row:
            raise KeyError(
                f"Attributo mancante: {attribute}"
            )

        value = int(row[attribute])

        if not (
            0
            <= value
            < EXPECTED_VARIANTS[attribute]
        ):
            raise ValueError(
                f"Valore {value} non valido "
                f"per {attribute}."
            )

        fragments.append(
            f"{phrase} {value}"
        )

    if len(fragments) == 1:
        description = fragments[0]
    else:
        description = (
            ", ".join(fragments[:-1])
            + ", and "
            + fragments[-1]
        )

    return f"{prefix} {description}"


def build_captions(
    assigned: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    required_columns = {
        "image_id",
        "image_path",
        "split",
    }

    caption_attributes = {
        item["key"]
        for item in config[
            "caption"
        ]["attributes"]
    }

    required_columns |= caption_attributes

    missing_columns = (
        required_columns
        - set(assigned.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonne mancanti: "
            f"{sorted(missing_columns)}"
        )

    if assigned["image_id"].duplicated().any():
        raise ValueError(
            "Sono presenti image_id duplicati."
        )

    found_splits = set(
        assigned["split"].unique()
    )

    if found_splits != set(VALID_SPLITS):
        raise ValueError(
            f"Split trovati: {found_splits}; "
            f"attesi: {set(VALID_SPLITS)}."
        )

    output = (
        assigned
        .sort_values("image_id")
        .reset_index(drop=True)
        .copy()
    )

    output["caption"] = [
        generate_caption(
            row=row,
            caption_config=config["caption"],
        )
        for row in output.to_dict(
            orient="records"
        )
    ]

    if output["caption"].isna().any():
        raise RuntimeError(
            "Sono state generate caption mancanti."
        )

    if (
        output["caption"]
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Sono state generate caption vuote."
        )

    return output


def save_captions(
    captioned: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        output_dir
        / "metadata_with_captions.csv"
    )

    captioned.to_csv(
        full_path,
        index=False,
    )

    split_counts = {}

    for split_name in VALID_SPLITS:
        current = captioned[
            captioned["split"] == split_name
        ].copy()

        split_path = (
            output_dir
            / f"{split_name}.csv"
        )

        current.to_csv(
            split_path,
            index=False,
        )

        split_counts[split_name] = int(
            len(current)
        )

    caption_lengths = (
        captioned["caption"]
        .str.split()
        .str.len()
    )

    summary = {
        "num_examples": int(
            len(captioned)
        ),
        "split_counts": split_counts,
        "caption_attributes": [
            item["key"]
            for item in config[
                "caption"
            ]["attributes"]
        ],
        "caption_prefix": config[
            "caption"
        ]["prefix"],
        "minimum_whitespace_length": int(
            caption_lengths.min()
        ),
        "maximum_whitespace_length": int(
            caption_lengths.max()
        ),
        "num_unique_captions": int(
            captioned[
                "caption"
            ].nunique()
        ),
        "captions_sha256": compute_sha256(
            full_path
        ),
    }

    summary_path = (
        output_dir
        / "caption_summary.json"
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
        "--splits",
        type=Path,
        default=Path(
            "data/processed/splits/"
            "metadata_with_splits.csv"
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
            "data/processed/captions"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assigned = pd.read_csv(args.splits)
    config = load_text_config(args.config)

    captioned = build_captions(
        assigned=assigned,
        config=config,
    )

    summary = save_captions(
        captioned=captioned,
        output_dir=args.output_dir,
        config=config,
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
