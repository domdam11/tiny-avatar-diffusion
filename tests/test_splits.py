from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.data.build_splits import (
    VALID_SPLITS,
    assign_splits,
    load_split_config,
    validate_assignments,
)


METADATA_PATH = Path("data/processed/metadata.csv")
CONFIG_PATH = Path("configs/data_split.yaml")
SPLITS_PATH = Path(
    "data/processed/splits/metadata_with_splits.csv"
)


def load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    # Carica gli input condivisi dai test
    assert METADATA_PATH.exists()
    assert CONFIG_PATH.exists()
    assert SPLITS_PATH.exists()

    metadata = pd.read_csv(METADATA_PATH)
    config = load_split_config(CONFIG_PATH)
    assigned = pd.read_csv(SPLITS_PATH)

    return metadata, config, assigned


def test_total_number_of_examples() -> None:
    # Lo split deve contenere tutti i 10.000 esempi
    metadata, _, assigned = load_inputs()

    assert len(metadata) == 10_000
    assert len(assigned) == 10_000


def test_unique_assignments() -> None:
    # Ogni immagine deve essere assegnata una sola volta
    _, _, assigned = load_inputs()

    assert assigned["image_id"].nunique() == 10_000
    assert not assigned["image_id"].duplicated().any()


def test_expected_split_names() -> None:
    # Devono essere presenti esattamente i quattro split previsti
    _, _, assigned = load_inputs()

    assert set(assigned["split"].unique()) == set(
        VALID_SPLITS
    )


def test_expected_split_sizes() -> None:
    # Le cardinalità attese rendono lo split facilmente verificabile
    _, _, assigned = load_inputs()

    counts = assigned["split"].value_counts()

    assert int(counts["train"]) == 7_884
    assert int(counts["validation"]) == 986
    assert int(counts["test_ordinary"]) == 986
    assert int(counts["test_ood"]) == 144


def test_ood_combination_is_held_out() -> None:
    # La combinazione OOD non deve comparire negli split ordinari
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

    ordinary_mask = assigned["split"] != "test_ood"

    assert not (
        (
            assigned.loc[
                ordinary_mask,
                "hair_color",
            ]
            == 3
        )
        & (
            assigned.loc[
                ordinary_mask,
                "face_shape",
            ]
            == 3
        )
    ).any()


def test_ood_contains_only_selected_combination() -> None:
    # test_ood non deve includere esempi non appartenenti alla coppia
    _, _, assigned = load_inputs()

    test_ood = assigned[
        assigned["split"] == "test_ood"
    ]

    assert (test_ood["hair_color"] == 3).all()
    assert (test_ood["face_shape"] == 3).all()


def test_individual_values_remain_in_training() -> None:
    # I valori singoli devono restare osservabili durante il training
    _, _, assigned = load_inputs()

    train = assigned[
        assigned["split"] == "train"
    ]

    assert (train["hair_color"] == 3).any()
    assert (train["face_shape"] == 3).any()


def test_split_is_deterministic() -> None:
    # Stesso input e stesso seed devono produrre lo stesso split
    metadata, config, _ = load_inputs()

    first_assignment = assign_splits(
        metadata=metadata,
        config=config,
    )

    second_assignment = assign_splits(
        metadata=metadata,
        config=config,
    )

    pdt.assert_frame_equal(
        first_assignment,
        second_assignment,
        check_like=False,
    )
