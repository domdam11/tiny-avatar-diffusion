
import math
from pathlib import Path

import pytest
import torch

from tokenizers import Tokenizer

from src.data.avatar_dataset import (
    VALID_SPLITS,
    build_dataloader,
    build_datasets,
    image_to_tensor,
    load_data_config,
)
from src.data.build_metadata import (
    EXPECTED_VARIANTS,
)
from src.diffusion.schedule import (
    DDPMSchedule,
    load_diffusion_config,
)


DATA_CONFIG_PATH = Path(
    "configs/data.yaml"
)

DIFFUSION_CONFIG_PATH = Path(
    "configs/diffusion.yaml"
)


@pytest.fixture(scope="module")
def data_config():
    return load_data_config(
        DATA_CONFIG_PATH
    )


@pytest.fixture(scope="module")
def datasets(data_config):
    return build_datasets(
        data_config
    )


def test_split_lengths(
    data_config,
    datasets,
):
    expected_counts = data_config[
        "data"
    ]["expected_split_counts"]

    for split_name in VALID_SPLITS:
        assert (
            len(datasets[split_name])
            == expected_counts[split_name]
        )

    assert sum(
        len(dataset)
        for dataset in datasets.values()
    ) == 10_000


def test_dataset_sequence_length(
    datasets,
):
    for dataset in datasets.values():
        assert dataset.sequence_length == 39


def test_sample_structure(
    datasets,
):
    sample = datasets["train"][0]

    expected_keys = {
        "image",
        "input_ids",
        "attention_mask",
        "attributes",
        "conditioning_values",
        "image_id",
        "caption",
        "image_path",
        "split",
    }

    assert set(sample) == expected_keys


def test_image_tensor(
    datasets,
):
    sample = datasets["train"][0]
    image = sample["image"]

    assert image.shape == (
        3,
        32,
        32,
    )

    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert image.min() >= -1.0
    assert image.max() <= 1.0


def test_image_loading_is_deterministic(
    datasets,
):
    dataset = datasets["train"]

    first = dataset[0]["image"]
    second = dataset[0]["image"]

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_text_tensors(
    datasets,
):
    sample = datasets["train"][0]

    assert sample["input_ids"].shape == (
        39,
    )

    assert (
        sample["attention_mask"].shape
        == (39,)
    )

    assert (
        sample["input_ids"].dtype
        == torch.long
    )

    assert (
        sample["attention_mask"].dtype
        == torch.long
    )

    assert set(
        sample[
            "attention_mask"
        ].tolist()
    ).issubset({0, 1})


def test_tokens_match_saved_tokenizer(
    data_config,
    datasets,
):
    tokenizer = Tokenizer.from_file(
        data_config[
            "data"
        ]["tokenizer_path"]
    )

    sample = datasets["train"][0]

    encoding = tokenizer.encode(
        sample["caption"]
    )

    assert (
        sample["input_ids"].tolist()
        == encoding.ids
    )

    assert (
        sample[
            "attention_mask"
        ].tolist()
        == encoding.attention_mask
    )


def test_attribute_tensors(
    data_config,
    datasets,
):
    sample = datasets["train"][0]

    conditioning_attributes = (
        data_config[
            "data"
        ]["conditioning_attributes"]
    )

    assert sample["attributes"].shape == (
        len(EXPECTED_VARIANTS),
    )

    assert (
        sample[
            "conditioning_values"
        ].shape
        == (
            len(conditioning_attributes),
        )
    )

    assert (
        sample["attributes"].dtype
        == torch.long
    )

    assert (
        sample[
            "conditioning_values"
        ].dtype
        == torch.long
    )


def test_split_labels(
    datasets,
):
    for split_name in VALID_SPLITS:
        dataset = datasets[split_name]

        assert dataset[0]["split"] == (
            split_name
        )

        assert dataset[-1]["split"] == (
            split_name
        )


def test_split_ids_are_disjoint(
    datasets,
):
    id_sets = {
        split_name: set(
            dataset.image_ids
        )
        for split_name, dataset
        in datasets.items()
    }

    for first_index, first_name in enumerate(
        VALID_SPLITS
    ):
        for second_name in VALID_SPLITS[
            first_index + 1:
        ]:
            assert id_sets[
                first_name
            ].isdisjoint(
                id_sets[second_name]
            )


def test_default_batch_collation(
    datasets,
):
    loader = build_dataloader(
        dataset=datasets["train"],
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        seed=2026,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (
        8,
        3,
        32,
        32,
    )

    assert batch["input_ids"].shape == (
        8,
        39,
    )

    assert (
        batch["attention_mask"].shape
        == (8, 39)
    )

    assert batch["attributes"].shape == (
        8,
        len(EXPECTED_VARIANTS),
    )

    assert (
        batch[
            "conditioning_values"
        ].shape[0]
        == 8
    )

    assert len(batch["image_id"]) == 8
    assert len(batch["caption"]) == 8


def test_training_shuffle_is_reproducible(
    datasets,
):
    loader_arguments = {
        "dataset": datasets["train"],
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "drop_last": False,
        "seed": 2026,
    }

    first_loader = build_dataloader(
        **loader_arguments
    )

    second_loader = build_dataloader(
        **loader_arguments
    )

    first_batch = next(
        iter(first_loader)
    )

    second_batch = next(
        iter(second_loader)
    )

    assert (
        first_batch["image_id"]
        == second_batch["image_id"]
    )

    torch.testing.assert_close(
        first_batch["image"],
        second_batch["image"],
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        first_batch["input_ids"],
        second_batch["input_ids"],
        rtol=0.0,
        atol=0.0,
    )


def test_evaluation_loader_is_sequential(
    datasets,
):
    dataset = datasets["validation"]

    loader = build_dataloader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        seed=2026,
    )

    batch = next(iter(loader))

    assert batch["image_id"] == (
        dataset.image_ids[:8]
    )


def test_no_examples_dropped(
    datasets,
):
    dataset = datasets["train"]

    loader = build_dataloader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        seed=2026,
    )

    assert len(loader) == math.ceil(
        len(dataset) / 64
    )

    loaded_ids = []

    for batch in loader:
        loaded_ids.extend(
            batch["image_id"]
        )

    assert loaded_ids == (
        dataset.image_ids
    )

    assert len(loaded_ids) == 7884


def test_batch_integrates_with_diffusion(
    datasets,
):
    loader = build_dataloader(
        dataset=datasets["train"],
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        seed=2026,
    )

    batch = next(iter(loader))
    images = batch["image"]

    diffusion_config = (
        load_diffusion_config(
            DIFFUSION_CONFIG_PATH
        )
    )

    schedule = (
        DDPMSchedule.from_config(
            diffusion_config
        )
    )

    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
    )

    noisy_images, noise = (
        schedule.q_sample(
            x_start=images,
            timesteps=timesteps,
        )
    )

    assert noisy_images.shape == (
        images.shape
    )

    assert noise.shape == images.shape
    assert torch.isfinite(
        noisy_images
    ).all()


def test_production_multiworker_batch(
    data_config,
    datasets,
):
    loader_config = data_config[
        "loader"
    ]

    loader = build_dataloader(
        dataset=datasets["train"],
        batch_size=8,
        shuffle=True,
        num_workers=int(
            loader_config["num_workers"]
        ),
        pin_memory=bool(
            loader_config["pin_memory"]
        ),
        persistent_workers=bool(
            loader_config[
                "persistent_workers"
            ]
        ),
        prefetch_factor=int(
            loader_config[
                "prefetch_factor"
            ]
        ),
        drop_last=False,
        seed=int(
            loader_config["seed"]
        ),
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (
        8,
        3,
        32,
        32,
    )

    assert batch["input_ids"].shape == (
        8,
        39,
    )
