import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from src.data.avatar_dataset import (
    build_dataloader,
    build_dataloaders,
    build_datasets,
    image_to_tensor,
)
from src.data.build_splits import VALID_SPLITS

CONFIG_PATH = Path("configs/data_loader.yaml")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_expected_sequence_length() -> int:
    summary_path = Path(
        "data/processed/tokenizer/tokenizer_summary.json"
    )
    assert summary_path.exists(), (
        f"Tokenizer summary non trovato: {summary_path}"
    )

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    return int(summary["sequence_length_with_boundaries"])


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def datasets(config: dict) -> dict:
    return build_datasets(config)


def test_all_splits_are_built(datasets: dict) -> None:
    assert set(datasets) == set(VALID_SPLITS)


def test_expected_split_sizes(
    datasets: dict,
    config: dict,
) -> None:
    expected_counts = config["data"]["expected_split_counts"]
    for split_name in VALID_SPLITS:
        assert len(datasets[split_name]) == int(
            expected_counts[split_name]
        )


def test_conditioning_attributes_are_consistent(
    datasets: dict,
    config: dict,
) -> None:
    expected_attributes = config["data"][
        "conditioning_attributes"
    ]
    dataset = datasets["train"]

    assert dataset.conditioning_attributes == tuple(
        expected_attributes
    )
    assert dataset.conditioning_values.shape[1] == len(
        expected_attributes
    )


def test_caption_tokens_have_uniform_length(
    datasets: dict,
) -> None:
    lengths = {
        dataset.sequence_length
        for dataset in datasets.values()
    }

    assert len(lengths) == 1
    assert next(iter(lengths)) == load_expected_sequence_length()


def test_images_are_normalized(
    datasets: dict,
) -> None:
    indices = [0, len(datasets["train"]) // 2, -1]
    for index in indices:
        image = datasets["train"][index]["image"]
        assert torch.isfinite(image).all()
        assert image.min() >= -1.0
        assert image.max() <= 1.0


def test_validation_loader_keeps_all_examples(
    datasets: dict,
    config: dict,
) -> None:
    loaders = build_dataloaders(
        datasets=datasets,
        config=config,
    )
    validation_count = sum(
        batch["image"].shape[0]
        for batch in loaders["validation"]
    )
    assert validation_count == len(datasets["validation"])


def test_train_dataloader_batch(
    datasets: dict,
    config: dict,
) -> None:
    loader_config = config["loader"]

    loader = build_dataloader(
        dataset=datasets["train"],
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=bool(loader_config["pin_memory"]),
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=True,
        seed=42,
    )

    batch = next(iter(loader))
    expected_sequence_length = (
        load_expected_sequence_length()
    )

    assert batch["image"].shape == (64, 3, 32, 32)
    assert batch["input_ids"].shape == (
        64,
        expected_sequence_length,
    )
    assert batch["attention_mask"].shape == (
        64,
        expected_sequence_length,
    )
    assert batch["attributes"].shape == (64, 18)
    assert batch["conditioning_values"].shape == (64, 8)

    assert batch["image"].dtype == torch.float32
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long


def test_evaluation_loader_keeps_final_batch(
    datasets: dict,
    config: dict,
) -> None:
    loaders = build_dataloaders(
        datasets=datasets,
        config=config,
    )
    expected_total = len(datasets["test_ood"])
    observed_total = sum(
        batch["image"].shape[0]
        for batch in loaders["test_ood"]
    )
    assert observed_total == expected_total


def test_train_sample_shapes_and_types(
    datasets: dict,
) -> None:
    sample = datasets["train"][0]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["image"].dtype == torch.float32

    assert sample["input_ids"].ndim == 1
    assert sample["input_ids"].dtype == torch.long

    assert (
        sample["attention_mask"].shape
        == sample["input_ids"].shape
    )
    assert sample["attention_mask"].dtype == torch.long

    assert sample["attributes"].shape == (18,)
    assert sample["conditioning_values"].shape == (8,)
    assert sample["split"] == "train"


def test_image_to_tensor_rejects_invalid_size() -> None:
    image = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError):
        image_to_tensor(image=image, image_size=0)
