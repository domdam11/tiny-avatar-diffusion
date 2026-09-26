
from pathlib import Path

import pandas as pd

from src.data.build_metadata import EXPECTED_VARIANTS


METADATA_PATH = Path(
    "data/processed/metadata.csv"
)


def load_metadata():
    assert METADATA_PATH.exists()
    return pd.read_csv(METADATA_PATH)


def test_number_of_examples():
    metadata = load_metadata()
    assert len(metadata) == 10_000


def test_unique_ids():
    metadata = load_metadata()
    assert metadata["image_id"].nunique() == 10_000
    assert not metadata["image_id"].duplicated().any()


def test_no_missing_values():
    metadata = load_metadata()
    assert not metadata.isna().any().any()


def test_attribute_columns():
    metadata = load_metadata()

    assert set(EXPECTED_VARIANTS).issubset(
        metadata.columns
    )


def test_attribute_ranges():
    metadata = load_metadata()

    for attribute, num_variants in (
        EXPECTED_VARIANTS.items()
    ):
        assert metadata[attribute].min() >= 0
        assert metadata[attribute].max() < num_variants


def test_image_and_csv_paths():
    metadata = load_metadata()
    dataset_root = Path("data/raw/cartoonset10k")

    for row in metadata.itertuples(index=False):
        assert (
            dataset_root / row.image_path
        ).exists()

        assert (
            dataset_root / row.csv_path
        ).exists()
