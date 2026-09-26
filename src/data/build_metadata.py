
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


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


def parse_attribute_file(csv_path: Path) -> dict:
    table = pd.read_csv(
        csv_path,
        header=None,
        names=["attribute", "value", "num_variants"],
        skipinitialspace=True,
    )

    table["attribute"] = (
        table["attribute"]
        .astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
    )

    if table["attribute"].duplicated().any():
        duplicates = table.loc[
            table["attribute"].duplicated(),
            "attribute",
        ].tolist()
        raise ValueError(
            f"Attributi duplicati in {csv_path}: {duplicates}"
        )

    found_attributes = set(table["attribute"])
    expected_attributes = set(EXPECTED_VARIANTS)

    missing = sorted(expected_attributes - found_attributes)
    unexpected = sorted(found_attributes - expected_attributes)

    if missing or unexpected:
        raise ValueError(
            f"Schema non valido in {csv_path}. "
            f"Mancanti={missing}; inattesi={unexpected}"
        )

    output = {}

    for row in table.itertuples(index=False):
        attribute = row.attribute
        value = int(row.value)
        num_variants = int(row.num_variants)
        expected_num_variants = EXPECTED_VARIANTS[attribute]

        if num_variants != expected_num_variants:
            raise ValueError(
                f"{csv_path}: {attribute} dichiara "
                f"{num_variants} varianti, attese "
                f"{expected_num_variants}."
            )

        if not 0 <= value < num_variants:
            raise ValueError(
                f"{csv_path}: valore {value} non valido "
                f"per {attribute}, che ha {num_variants} varianti."
            )

        output[attribute] = value

    return output


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_metadata(
    dataset_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(dataset_root.rglob("*.csv"))
    png_files = sorted(dataset_root.rglob("*.png"))

    if not csv_files:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {dataset_root}"
        )

    if not png_files:
        raise FileNotFoundError(
            f"Nessun PNG trovato in {dataset_root}"
        )

    png_by_stem = {}

    for image_path in png_files:
        if image_path.stem in png_by_stem:
            raise ValueError(
                f"ID immagine duplicato: {image_path.stem}"
            )
        png_by_stem[image_path.stem] = image_path

    rows = []
    seen_ids = set()

    for csv_path in csv_files:
        image_id = csv_path.stem

        if image_id in seen_ids:
            raise ValueError(
                f"ID CSV duplicato: {image_id}"
            )

        seen_ids.add(image_id)

        if image_id not in png_by_stem:
            raise FileNotFoundError(
                f"PNG mancante per {csv_path}"
            )

        image_path = png_by_stem[image_id]
        attributes = parse_attribute_file(csv_path)

        row = {
            "image_id": image_id,
            "image_path": image_path.relative_to(
                dataset_root
            ).as_posix(),
            "csv_path": csv_path.relative_to(
                dataset_root
            ).as_posix(),
            **attributes,
        }

        rows.append(row)

    png_without_csv = sorted(
        set(png_by_stem) - seen_ids
    )

    if png_without_csv:
        raise ValueError(
            f"Trovati {len(png_without_csv)} PNG senza CSV. "
            f"Primi ID: {png_without_csv[:5]}"
        )

    metadata = pd.DataFrame(rows)
    metadata = metadata.sort_values(
        "image_id"
    ).reset_index(drop=True)

    if metadata["image_id"].duplicated().any():
        raise ValueError(
            "Il metadata finale contiene ID duplicati."
        )

    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "num_examples": int(len(metadata)),
        "num_attributes": len(EXPECTED_VARIANTS),
        "attributes": EXPECTED_VARIANTS,
        "metadata_filename": metadata_path.name,
        "metadata_sha256": compute_file_sha256(
            metadata_path
        ),
    }

    summary_path = output_dir / "metadata_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    return metadata


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    metadata = build_metadata(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )

    print(f"Esempi elaborati: {len(metadata)}")
    print(
        f"Attributi elaborati: "
        f"{len(EXPECTED_VARIANTS)}"
    )
    print(
        f"Metadata salvato in: "
        f"{args.output_dir / 'metadata.csv'}"
    )
    print(
        f"Riepilogo salvato in: "
        f"{args.output_dir / 'metadata_summary.json'}"
    )


if __name__ == "__main__":
    main()
