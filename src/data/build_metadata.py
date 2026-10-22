import argparse
import json
from pathlib import Path

import pandas as pd


# Numero atteso di varianti per ciascun attributo discreto.
EXPECTED_VARIANTS = {
    "eye_angle": 3,
    "eye_lashes": 2,
    "eye_lid": 2,
    "chin_length": 3,
    "eyebrow_weight": 2,
    "eyebrow_shape": 14,
    "eyebrow_thickness": 4,
    "face_shape": 7,
    "facial_hair": 15,
    "hair": 111,
    "eye_color": 5,
    "face_color": 11,
    "hair_color": 10,
    "glasses": 12,
    "glasses_color": 7,
    "eye_slant": 3,
    "eyebrow_width": 3,
    "eye_eyebrow_distance": 3,
}


def parse_single_metadata(csv_path: Path, dataset_root: Path) -> dict:
    # Ogni CSV descrive un singolo avatar come tabella verticale
    # attributo -> valore -> numero totale di varianti possibili.
    raw = pd.read_csv(
        csv_path,
        header=None,
        names=["attribute", "value", "num_variants"],
    )

    # Verifichiamo che gli attributi siano esattamente quelli attesi.
    observed_attributes = raw["attribute"].tolist()
    expected_attributes = list(EXPECTED_VARIANTS.keys())

    if observed_attributes != expected_attributes:
        raise ValueError(
            f"Ordine o contenuto attributi non valido in {csv_path.name}."
        )

    # Verifichiamo che il numero di varianti dichiarato nel CSV
    # coincida con quello atteso per ogni attributo.
    observed_num_variants = {
        row.attribute: int(row.num_variants)
        for row in raw.itertuples(index=False)
    }

    if observed_num_variants != EXPECTED_VARIANTS:
        raise ValueError(
            f"Numero varianti non coerente in {csv_path.name}."
        )

    image_id = csv_path.stem
    image_path = csv_path.with_suffix(".png")

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Immagine mancante per {image_id}."
        )

    # Rappresentazione piatta: una riga per immagine,
    # una colonna per attributo.
    record = {
        "image_id": image_id,
        "image_path": str(image_path.relative_to(dataset_root)),
        "csv_path": str(csv_path.relative_to(dataset_root)),
    }

    for row in raw.itertuples(index=False):
        record[row.attribute] = int(row.value)

    return record


def build_metadata(dataset_root: Path) -> pd.DataFrame:
    # Costruiamo il metadata centralizzato iterando in modo deterministico
    # sui CSV ordinati lessicograficamente.
    csv_paths = sorted(dataset_root.rglob("*.csv"))

    if len(csv_paths) != 10_000:
        raise ValueError(
            f"Attesi 10.000 CSV, trovati {len(csv_paths)}."
        )

    records = [
        parse_single_metadata(csv_path, dataset_root)
        for csv_path in csv_paths
    ]

    metadata = pd.DataFrame(records)

    # Ordinamento canonico per riproducibilità.
    metadata = (
        metadata
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    if metadata["image_id"].duplicated().any():
        raise ValueError("Sono presenti image_id duplicati.")

    return metadata


def build_summary(metadata: pd.DataFrame) -> dict:
    # Riepilogo semplice ma utile per i controlli successivi.
    return {
        "num_examples": int(len(metadata)),
        "num_attributes": int(len(EXPECTED_VARIANTS)),
        "columns": list(metadata.columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/cartoonset10k"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/metadata.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/metadata_summary.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(args.dataset_root)
    summary = build_summary(metadata)

    metadata.to_csv(args.output, index=False)

    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
