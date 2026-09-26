
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.build_metadata import EXPECTED_VARIANTS


CANDIDATE_PAIRS = [
    ("hair_color", "face_shape"),
    ("eye_color", "chin_length"),
    ("hair_color", "chin_length"),
    ("eye_color", "face_shape"),
    ("glasses", "face_shape"),
]


def build_attribute_counts(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    tables = []

    for attribute in EXPECTED_VARIANTS:
        counts = (
            metadata[attribute]
            .value_counts()
            .sort_index()
            .rename_axis("value")
            .reset_index(name="count")
        )

        counts.insert(0, "attribute", attribute)
        counts["frequency"] = (
            counts["count"] / len(metadata)
        )

        tables.append(counts)

    return pd.concat(
        tables,
        ignore_index=True,
    )


def build_pair_counts(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    tables = []

    for first, second in CANDIDATE_PAIRS:
        counts = (
            metadata.groupby(
                [first, second],
                observed=True,
            )
            .size()
            .reset_index(name="count")
        )

        counts.insert(
            0,
            "pair",
            f"{first}__{second}",
        )

        counts = counts.rename(
            columns={
                first: "first_value",
                second: "second_value",
            }
        )

        counts["frequency"] = (
            counts["count"] / len(metadata)
        )

        tables.append(counts)

    return pd.concat(
        tables,
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/processed/metadata.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/audit"
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(args.metadata)

    attribute_counts = build_attribute_counts(
        metadata
    )

    pair_counts = build_pair_counts(
        metadata
    )

    attribute_counts.to_csv(
        args.output_dir / "attribute_counts.csv",
        index=False,
    )

    pair_counts.to_csv(
        args.output_dir / "candidate_pair_counts.csv",
        index=False,
    )

    summary = {
        "num_examples": int(len(metadata)),
        "attributes": {},
        "candidate_pairs": {},
    }

    for attribute, num_variants in (
        EXPECTED_VARIANTS.items()
    ):
        counts = metadata[attribute].value_counts()

        summary["attributes"][attribute] = {
            "expected_num_variants": num_variants,
            "observed_num_variants": int(
                counts.size
            ),
            "minimum_count": int(counts.min()),
            "maximum_count": int(counts.max()),
        }

    for first, second in CANDIDATE_PAIRS:
        name = f"{first}__{second}"

        selected = pair_counts[
            pair_counts["pair"] == name
        ]

        summary["candidate_pairs"][name] = {
            "observed_combinations": int(
                len(selected)
            ),
            "minimum_count": int(
                selected["count"].min()
            ),
            "maximum_count": int(
                selected["count"].max()
            ),
            "median_count": float(
                selected["count"].median()
            ),
        }

    with (
        args.output_dir / "audit_summary.json"
    ).open("w", encoding="utf-8") as file:
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

    for pair_name in pair_counts["pair"].unique():
        selected = (
            pair_counts[
                pair_counts["pair"] == pair_name
            ]
            .sort_values(
                "count",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        print(f"\n=== {pair_name} ===")
        print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
