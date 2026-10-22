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
    # Calcola il checksum SHA-256 di un file
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_text_config(config_path: Path) -> dict:
    # Carica e controlla la configurazione delle caption
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "caption" not in config or "tokenizer" not in config:
        raise ValueError(
            "Sezioni 'caption' o 'tokenizer' mancanti."
        )

    caption_config = config["caption"]

    if not caption_config.get("prefix"):
        raise ValueError("Prefix della caption mancante.")

    attributes = caption_config.get("attributes", [])

    if not attributes:
        raise ValueError("Nessun attributo configurato.")

    keys = [item["key"] for item in attributes]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Attributi duplicati nella configurazione."
        )

    unknown_attributes = sorted(
        set(keys) - set(EXPECTED_VARIANTS)
    )

    if unknown_attributes:
        raise ValueError(
            f"Attributi sconosciuti: {unknown_attributes}"
        )

    return config


def generate_caption(
    row: Mapping,
    caption_config: dict,
) -> str:
    # Genera una caption deterministica a partire dagli attributi
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
            0 <= value < EXPECTED_VARIANTS[attribute]
        ):
            raise ValueError(
                f"Valore {value} non valido per {attribute}."
            )

        # Il valore numerico mantiene il collegamento esplicito
        # tra testo e categoria originale del metadata.
        fragments.append(f"{phrase} {value}")

    # L'ultima congiunzione rende la caption grammaticalmente coerente
    # senza introdurre variazioni casuali nel template.
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
    # Aggiunge una caption a ogni riga del metadata con split
    required_columns = {
        "image_id",
        "image_path",
        "split",
    }

    caption_attributes = {
        item["key"]
        for item in config["caption"]["attributes"]
    }

    required_columns |= caption_attributes

    missing_columns = (
        required_columns - set(assigned.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Colonne mancanti: {sorted(missing_columns)}"
        )

    if assigned["image_id"].duplicated().any():
        raise ValueError("Sono presenti image_id duplicati.")

    # L'ordinamento per image_id mantiene l'output stabile
    # anche se il file di input viene riordinato.
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
        for row in output.to_dict(orient="records")
    ]

    if (
        output["caption"].isna().any()
        or output["caption"].str.strip().eq("").any()
    ):
        raise RuntimeError(
            "Generate caption mancanti o vuote."
        )

    return output


def save_captions(
    captioned: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> dict:
    # Salva caption complete, CSV per split e riepilogo JSON
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "metadata_with_captions.csv"

    # File centrale da usare in seguito per creare dataset e dataloader.
    captioned.to_csv(full_path, index=False)

    split_counts = {}

    for split_name in VALID_SPLITS:
        current = captioned[
            captioned["split"] == split_name
        ].copy()

        current.to_csv(
            output_dir / f"{split_name}.csv",
            index=False,
        )

        split_counts[split_name] = int(len(current))

    # La lunghezza whitespace è un controllo semplice sul template.
    # In questo progetto ogni caption ha struttura e lunghezza costanti.
    caption_lengths = (
        captioned["caption"]
        .str.split()
        .str.len()
    )

    summary = {
        "num_examples": int(len(captioned)),
        "split_counts": split_counts,
        "caption_attributes": [
            item["key"]
            for item in config["caption"]["attributes"]
        ],
        "caption_prefix": config["caption"]["prefix"],
        "minimum_whitespace_length": int(
            caption_lengths.min()
        ),
        "maximum_whitespace_length": int(
            caption_lengths.max()
        ),
        "num_unique_captions": int(
            captioned["caption"].nunique()
        ),
        "captions_sha256": compute_sha256(full_path),
    }

    summary_path = output_dir / "caption_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    return summary


def parse_args() -> argparse.Namespace:
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
        default=Path("data/processed/captions"),
    )

    return parser.parse_args()


def main() -> None:
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
