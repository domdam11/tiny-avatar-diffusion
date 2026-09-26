
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.build_metadata import EXPECTED_VARIANTS


VALID_SPLITS = (
    "train",
    "validation",
    "test_ordinary",
    "test_ood",
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


def load_split_config(config_path: Path) -> dict:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    required_sections = {
        "seed",
        "compositional_ood",
        "ordinary_split",
    }

    missing_sections = (
        required_sections - set(config)
    )

    if missing_sections:
        raise ValueError(
            "Sezioni mancanti nella configurazione: "
            f"{sorted(missing_sections)}"
        )

    ratios = config["ordinary_split"]

    ratio_sum = (
        float(ratios["train_ratio"])
        + float(ratios["validation_ratio"])
        + float(ratios["test_ratio"])
    )

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "Le proporzioni degli split ordinari "
            f"devono sommare a 1, non {ratio_sum}."
        )

    return config


def validate_metadata(
    metadata: pd.DataFrame,
    config: dict,
) -> None:
    ood_config = config["compositional_ood"]

    required_columns = {
        "image_id",
        "image_path",
        ood_config["first_attribute"],
        ood_config["second_attribute"],
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel metadata: "
            f"{sorted(missing_columns)}"
        )

    if metadata.empty:
        raise ValueError(
            "Il metadata è vuoto."
        )

    if metadata["image_id"].isna().any():
        raise ValueError(
            "Sono presenti image_id mancanti."
        )

    if metadata["image_id"].duplicated().any():
        duplicate_ids = metadata.loc[
            metadata["image_id"].duplicated(),
            "image_id",
        ].tolist()

        raise ValueError(
            "Sono presenti image_id duplicati: "
            f"{duplicate_ids[:5]}"
        )


def make_stratum(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    return dataframe[columns].astype(str).agg(
        "__".join,
        axis=1,
    )


def assign_splits(
    metadata: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    validate_metadata(metadata, config)

    metadata = (
        metadata
        .sort_values("image_id")
        .reset_index(drop=True)
        .copy()
    )

    seed = int(config["seed"])
    ood_config = config["compositional_ood"]
    ordinary_config = config["ordinary_split"]

    first_attribute = (
        ood_config["first_attribute"]
    )
    second_attribute = (
        ood_config["second_attribute"]
    )

    first_value = int(
        ood_config["first_value"]
    )
    second_value = int(
        ood_config["second_value"]
    )

    expected_ood_count = int(
        ood_config["expected_count"]
    )

    ood_mask = (
        (
            metadata[first_attribute]
            == first_value
        )
        & (
            metadata[second_attribute]
            == second_value
        )
    )

    ood = metadata.loc[ood_mask].copy()
    ordinary = metadata.loc[~ood_mask].copy()

    if len(ood) != expected_ood_count:
        raise ValueError(
            "Numero inatteso di esempi OOD: "
            f"trovati {len(ood)}, "
            f"attesi {expected_ood_count}."
        )

    if ordinary.empty:
        raise ValueError(
            "Lo split ordinario è vuoto."
        )

    stratify_columns = list(
        ordinary_config["stratify_by"]
    )

    ordinary["_stratum"] = make_stratum(
        ordinary,
        stratify_columns,
    )

    train_ratio = float(
        ordinary_config["train_ratio"]
    )
    validation_ratio = float(
        ordinary_config["validation_ratio"]
    )
    test_ratio = float(
        ordinary_config["test_ratio"]
    )

    temporary_ratio = (
        validation_ratio + test_ratio
    )

    train, temporary = train_test_split(
        ordinary,
        train_size=train_ratio,
        test_size=temporary_ratio,
        random_state=seed,
        shuffle=True,
        stratify=ordinary["_stratum"],
    )

    relative_test_ratio = (
        test_ratio / temporary_ratio
    )

    validation, test_ordinary = (
        train_test_split(
            temporary,
            test_size=relative_test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=temporary["_stratum"],
        )
    )

    split_frames = []

    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test_ordinary", test_ordinary),
        ("test_ood", ood),
    ):
        current = frame.drop(
            columns=["_stratum"],
            errors="ignore",
        ).copy()

        current.insert(
            1,
            "split",
            split_name,
        )

        split_frames.append(current)

    result = pd.concat(
        split_frames,
        ignore_index=True,
    )

    result = (
        result
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    if len(result) != len(metadata):
        raise RuntimeError(
            "Il numero finale di esempi non "
            "coincide con il metadata originale."
        )

    if result["image_id"].duplicated().any():
        raise RuntimeError(
            "Gli split contengono image_id duplicati."
        )

    if result["split"].isna().any():
        raise RuntimeError(
            "Alcuni esempi non hanno uno split."
        )

    return result


def validate_assignments(
    assigned: pd.DataFrame,
    config: dict,
) -> None:
    found_splits = set(
        assigned["split"].unique()
    )

    if found_splits != set(VALID_SPLITS):
        raise ValueError(
            f"Split trovati: {found_splits}; "
            f"attesi: {set(VALID_SPLITS)}."
        )

    ood_config = config["compositional_ood"]

    first_attribute = (
        ood_config["first_attribute"]
    )
    second_attribute = (
        ood_config["second_attribute"]
    )

    first_value = int(
        ood_config["first_value"]
    )
    second_value = int(
        ood_config["second_value"]
    )

    combination_mask = (
        (
            assigned[first_attribute]
            == first_value
        )
        & (
            assigned[second_attribute]
            == second_value
        )
    )

    combination_splits = set(
        assigned.loc[
            combination_mask,
            "split",
        ].unique()
    )

    if combination_splits != {"test_ood"}:
        raise ValueError(
            "La combinazione OOD compare negli "
            f"split sbagliati: {combination_splits}"
        )

    test_ood = assigned[
        assigned["split"] == "test_ood"
    ]

    if not combination_mask.loc[
        test_ood.index
    ].all():
        raise ValueError(
            "test_ood contiene esempi estranei "
            "alla combinazione selezionata."
        )

    train = assigned[
        assigned["split"] == "train"
    ]

    if first_value not in set(
        train[first_attribute]
    ):
        raise ValueError(
            f"{first_attribute}={first_value} "
            "non è presente nel training."
        )

    if second_value not in set(
        train[second_attribute]
    ):
        raise ValueError(
            f"{second_attribute}={second_value} "
            "non è presente nel training."
        )


def build_summary(
    assigned: pd.DataFrame,
    config: dict,
    assignments_path: Path,
) -> dict:
    ood_config = config["compositional_ood"]

    first_attribute = (
        ood_config["first_attribute"]
    )
    second_attribute = (
        ood_config["second_attribute"]
    )

    first_value = int(
        ood_config["first_value"]
    )
    second_value = int(
        ood_config["second_value"]
    )

    counts = (
        assigned["split"]
        .value_counts()
        .reindex(VALID_SPLITS)
    )

    train = assigned[
        assigned["split"] == "train"
    ]

    attribute_coverage = {}

    for attribute in EXPECTED_VARIANTS:
        expected_values = list(
            range(EXPECTED_VARIANTS[attribute])
        )

        observed_values = sorted(
            int(value)
            for value in train[
                attribute
            ].unique()
        )

        attribute_coverage[attribute] = {
            "expected_values": expected_values,
            "train_values": observed_values,
            "complete": (
                expected_values == observed_values
            ),
        }

    summary = {
        "seed": int(config["seed"]),
        "num_examples": int(len(assigned)),
        "counts": {
            split_name: int(counts[split_name])
            for split_name in VALID_SPLITS
        },
        "compositional_ood": {
            "first_attribute": first_attribute,
            "first_value": first_value,
            "second_attribute": second_attribute,
            "second_value": second_value,
            "count": int(
                counts["test_ood"]
            ),
        },
        "selected_values_in_training": {
            f"{first_attribute}={first_value}": int(
                (
                    train[first_attribute]
                    == first_value
                ).sum()
            ),
            f"{second_attribute}={second_value}": int(
                (
                    train[second_attribute]
                    == second_value
                ).sum()
            ),
        },
        "training_attribute_coverage": (
            attribute_coverage
        ),
        "assignments_sha256": compute_sha256(
            assignments_path
        ),
    }

    return summary


def save_splits(
    assigned: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        output_dir / "metadata_with_splits.csv"
    )

    assignments_path = (
        output_dir / "split_assignments.csv"
    )

    assigned.to_csv(
        full_path,
        index=False,
    )

    assigned[
        ["image_id", "split"]
    ].to_csv(
        assignments_path,
        index=False,
    )

    for split_name in VALID_SPLITS:
        split_path = (
            output_dir / f"{split_name}.csv"
        )

        assigned[
            assigned["split"] == split_name
        ].to_csv(
            split_path,
            index=False,
        )

    summary = build_summary(
        assigned=assigned,
        config=config,
        assignments_path=assignments_path,
    )

    summary_path = (
        output_dir / "split_summary.json"
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
        "--metadata",
        type=Path,
        default=Path(
            "data/processed/metadata.csv"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/data_split.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/splits"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    metadata = pd.read_csv(args.metadata)
    config = load_split_config(args.config)

    assigned = assign_splits(
        metadata=metadata,
        config=config,
    )

    validate_assignments(
        assigned=assigned,
        config=config,
    )

    summary = save_splits(
        assigned=assigned,
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
