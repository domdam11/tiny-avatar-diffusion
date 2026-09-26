
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.data.build_splits import (
    VALID_SPLITS,
    assign_splits,
    load_split_config,
    validate_assignments,
)


METADATA_PATH = Path(
    "data/processed/metadata.csv"
)

CONFIG_PATH = Path(
    "configs/data_split.yaml"
)

SPLITS_PATH = Path(
    "data/processed/splits/"
    "metadata_with_splits.csv"
)


def load_inputs():
    metadata = pd.read_csv(METADATA_PATH)
    config = load_split_config(CONFIG_PATH)
    assigned = pd.read_csv(SPLITS_PATH)

    return metadata, config, assigned


def test_total_number_of_examples():
    metadata, _, assigned = load_inputs()

    assert len(metadata) == 10_000
    assert len(assigned) == 10_000


def test_unique_assignments():
    _, _, assigned = load_inputs()

    assert assigned["image_id"].nunique() == 10_000
    assert not assigned["image_id"].duplicated().any()


def test_expected_splits():
    _, _, assigned = load_inputs()

    assert set(
        assigned["split"].unique()
    ) == set(VALID_SPLITS)


def test_expected_split_sizes():
    _, _, assigned = load_inputs()

    counts = assigned["split"].value_counts()

    assert counts["train"] == 7_884
    assert counts["validation"] == 986
    assert counts["test_ordinary"] == 986
    assert counts["test_ood"] == 144


def test_ood_combination_is_held_out():
    _, config, assigned = load_inputs()

    validate_assignments(
        assigned=assigned,
        config=config,
    )

    ood_mask = (
        (assigned["hair_color"] == 3)
        & (assigned["face_shape"] == 3)
    )

    assert (
        assigned.loc[ood_mask, "split"]
        == "test_ood"
    ).all()

    assert not (
        assigned.loc[
            assigned["split"] != "test_ood"
        ]
        .eval(
            "hair_color == 3 and face_shape == 3"
        )
        .any()
    )


def test_ood_contains_only_selected_combination():
    _, _, assigned = load_inputs()

    ood = assigned[
        assigned["split"] == "test_ood"
    ]

    assert (ood["hair_color"] == 3).all()
    assert (ood["face_shape"] == 3).all()


def test_individual_values_remain_in_training():
    _, _, assigned = load_inputs()

    train = assigned[
        assigned["split"] == "train"
    ]

    assert (train["hair_color"] == 3).any()
    assert (train["face_shape"] == 3).any()


def test_every_ordinary_pair_is_represented():
    _, _, assigned = load_inputs()

    expected_pairs = set(
        assigned.loc[
            assigned["split"] != "test_ood",
            ["hair_color", "face_shape"],
        ].itertuples(
            index=False,
            name=None,
        )
    )

    for split_name in (
        "train",
        "validation",
        "test_ordinary",
    ):
        split_pairs = set(
            assigned.loc[
                assigned["split"] == split_name,
                ["hair_color", "face_shape"],
            ].itertuples(
                index=False,
                name=None,
            )
        )

        assert split_pairs == expected_pairs


def test_split_is_deterministic():
    metadata, config, _ = load_inputs()

    first = assign_splits(
        metadata=metadata,
        config=config,
    )

    second = assign_splits(
        metadata=metadata,
        config=config,
    )

    pdt.assert_frame_equal(
        first,
        second,
        check_like=False,
    )
