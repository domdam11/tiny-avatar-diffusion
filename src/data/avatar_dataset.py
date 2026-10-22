import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.build_metadata import EXPECTED_VARIANTS
from src.data.build_splits import VALID_SPLITS


def image_to_tensor(
    image: Image.Image,
    image_size: int,
) -> Tensor:
    # Converte un'immagine PIL RGB in un tensore CHW in [-1, 1]
    if image_size < 1:
        raise ValueError(
            "image_size deve essere positivo."
        )

    image = image.convert("RGB")
    image = image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    # HWC [0, 255] -> CHW [-1, 1]
    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    )

    return tensor.div(127.5).sub(1.0)


class CartoonAvatarDataset(Dataset):
    # Dataset per immagini avatar e caption condizionali.

    # Le caption sono tokenizzate durante l'inizializzazione: questo evita
    # di ripetere la tokenizzazione ad ogni chiamata di __getitem__.


    def __init__(
        self,
        dataset_root: Path,
        captions_path: Path,
        tokenizer_path: Path,
        split: str,
        image_size: int,
        expected_count: Optional[int] = None,
        conditioning_attributes: Optional[
            Sequence[str]
        ] = None,
        validate_paths: bool = True,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Split non valido: {split}."
            )

        self.dataset_root = dataset_root.resolve()
        self.captions_path = captions_path
        self.tokenizer_path = tokenizer_path
        self.split = split
        self.image_size = int(image_size)

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root non trovata: "
                f"{self.dataset_root}"
            )

        if not self.captions_path.is_file():
            raise FileNotFoundError(
                f"File caption non trovato: "
                f"{self.captions_path}"
            )

        if not self.tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer non trovato: "
                f"{self.tokenizer_path}"
            )

        metadata = pd.read_csv(
            self.captions_path,
            dtype={
                "image_id": "string",
                "image_path": "string",
                "caption": "string",
                "split": "string",
            },
        )

        required_columns = {
            "image_id",
            "image_path",
            "caption",
            "split",
            *EXPECTED_VARIANTS.keys(),
        }

        missing_columns = (
            required_columns - set(metadata.columns)
        )

        if missing_columns:
            raise ValueError(
                "Colonne mancanti nel metadata: "
                f"{sorted(missing_columns)}"
            )

        if metadata["image_id"].duplicated().any():
            raise ValueError(
                "Il metadata contiene image_id duplicati."
            )

        available_splits = set(
            metadata["split"].unique()
        )

        if available_splits != set(VALID_SPLITS):
            raise ValueError(
                "Split trovati nel metadata: "
                f"{available_splits}; attesi: "
                f"{set(VALID_SPLITS)}."
            )

        # L'ordine canonico stabilizza dataset, test e campionamento.
        metadata = (
            metadata[
                metadata["split"] == split
            ]
            .sort_values("image_id")
            .reset_index(drop=True)
        )

        if metadata.empty:
            raise ValueError(
                f"Lo split {split} è vuoto."
            )

        if (
            expected_count is not None
            and len(metadata) != expected_count
        ):
            raise ValueError(
                f"Lo split {split} contiene "
                f"{len(metadata)} esempi; "
                f"attesi {expected_count}."
            )

        if metadata["caption"].isna().any():
            raise ValueError(
                "Sono presenti caption mancanti."
            )

        if (
            metadata["caption"]
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Sono presenti caption vuote."
            )

        if conditioning_attributes is None:
            conditioning_attributes = list(
                EXPECTED_VARIANTS
            )

        unknown_attributes = sorted(
            set(conditioning_attributes)
            - set(EXPECTED_VARIANTS)
        )

        if unknown_attributes:
            raise ValueError(
                "Attributi sconosciuti: "
                f"{unknown_attributes}"
            )

        self.conditioning_attributes = tuple(
            conditioning_attributes
        )

        self.all_attribute_names = tuple(
            EXPECTED_VARIANTS.keys()
        )

        image_paths = []

        for relative_path in metadata["image_path"]:
            absolute_path = (
                self.dataset_root
                / str(relative_path)
            ).resolve()

            # Protezione contro percorsi che escano dalla root del dataset.
            if (
                absolute_path != self.dataset_root
                and self.dataset_root
                not in absolute_path.parents
            ):
                raise ValueError(
                    "Percorso immagine esterno "
                    "al dataset root: "
                    f"{relative_path}"
                )

            image_paths.append(absolute_path)

        if validate_paths:
            missing_paths = [
                path
                for path in image_paths
                if not path.is_file()
            ]

            if missing_paths:
                raise FileNotFoundError(
                    f"Mancano {len(missing_paths)} "
                    "immagini. Primo percorso: "
                    f"{missing_paths[0]}"
                )

        tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )

        captions = (
            metadata["caption"]
            .astype(str)
            .tolist()
        )

        encodings = tokenizer.encode_batch(captions)

        encoded_lengths = {
            len(encoding.ids)
            for encoding in encodings
        }

        mask_lengths = {
            len(encoding.attention_mask)
            for encoding in encodings
        }

        if len(encoded_lengths) != 1:
            raise ValueError(
                "Le sequenze tokenizzate non hanno "
                "lunghezza uniforme."
            )

        if encoded_lengths != mask_lengths:
            raise ValueError(
                "Token ID e attention mask hanno "
                "lunghezze differenti."
            )

        self.sequence_length = next(
            iter(encoded_lengths)
        )

        self.input_ids = torch.tensor(
            [
                encoding.ids
                for encoding in encodings
            ],
            dtype=torch.long,
        )

        self.attention_masks = torch.tensor(
            [
                encoding.attention_mask
                for encoding in encodings
            ],
            dtype=torch.long,
        )

        self.all_attributes = torch.tensor(
            metadata[
                list(self.all_attribute_names)
            ].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

        self.conditioning_values = torch.tensor(
            metadata[
                list(self.conditioning_attributes)
            ].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

        self.image_ids = (
            metadata["image_id"]
            .astype(str)
            .tolist()
        )

        self.image_paths = image_paths
        self.captions = captions

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image_tensor = image_to_tensor(
                image=image,
                image_size=self.image_size,
            )

        if image_tensor.shape != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise RuntimeError(
                "Shape immagine inattesa: "
                f"{tuple(image_tensor.shape)}"
            )

        if not torch.isfinite(image_tensor).all():
            raise RuntimeError(
                f"Valori non finiti in {image_path}"
            )

        if (
            image_tensor.min() < -1.0
            or image_tensor.max() > 1.0
        ):
            raise RuntimeError(
                "Immagine fuori dall'intervallo "
                "[-1, 1]."
            )

        return {
            "image": image_tensor,
            "input_ids": self.input_ids[index],
            "attention_mask": (
                self.attention_masks[index]
            ),
            "attributes": self.all_attributes[index],
            "conditioning_values": (
                self.conditioning_values[index]
            ),
            "image_id": self.image_ids[index],
            "caption": self.captions[index],
            "image_path": str(image_path),
            "split": self.split,
        }


def seed_worker(worker_id: int) -> None:
    # Inizializza NumPy e random in ogni worker del DataLoader
    del worker_id

    worker_seed = (
        torch.initial_seed() % 2**32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: CartoonAvatarDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    # Costruisce un DataLoader riproducibile per uno split
    if batch_size < 1:
        raise ValueError(
            "batch_size deve essere positivo."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers non può essere negativo."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(
            pin_memory
            and torch.cuda.is_available()
        ),
        "drop_last": drop_last,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    # PyTorch non accetta questi argomenti con num_workers=0.
    if num_workers > 0:
        loader_arguments.update(
            {
                "persistent_workers": (
                    persistent_workers
                ),
                "prefetch_factor": (
                    prefetch_factor
                ),
            }
        )

    return DataLoader(**loader_arguments)


def build_datasets(
    config: dict,
) -> dict[str, CartoonAvatarDataset]:
    # Costruisce un dataset per ogni split del protocollo
    data_config = config["data"]

    dataset_root = Path(
        data_config["dataset_root"]
    )

    captions_path = Path(
        data_config["captions_path"]
    )

    tokenizer_path = Path(
        data_config["tokenizer_path"]
    )

    image_size = int(data_config["image_size"])

    expected_counts = data_config[
        "expected_split_counts"
    ]

    conditioning_attributes = data_config[
        "conditioning_attributes"
    ]

    datasets = {}

    for split_name in VALID_SPLITS:
        datasets[split_name] = CartoonAvatarDataset(
            dataset_root=dataset_root,
            captions_path=captions_path,
            tokenizer_path=tokenizer_path,
            split=split_name,
            image_size=image_size,
            expected_count=int(
                expected_counts[split_name]
            ),
            conditioning_attributes=(
                conditioning_attributes
            ),
        )

    return datasets


def build_dataloaders(
    datasets: dict[str, CartoonAvatarDataset],
    config: dict,
) -> dict[str, DataLoader]:
    # Costruisce un DataLoader per ogni split
    loader_config = config["loader"]

    loaders = {}

    for split_name in VALID_SPLITS:
        is_training = split_name == "train"

        batch_size = int(
            loader_config[
                "train_batch_size"
                if is_training
                else "evaluation_batch_size"
            ]
        )

        loaders[split_name] = build_dataloader(
            dataset=datasets[split_name],
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=int(
                loader_config["num_workers"]
            ),
            pin_memory=bool(
                loader_config["pin_memory"]
            ),
            persistent_workers=bool(
                loader_config["persistent_workers"]
            ),
            prefetch_factor=int(
                loader_config["prefetch_factor"]
            ),
            # I batch evaluation non devono perdere esempi.
            drop_last=bool(
                loader_config["drop_last"]
            )
            if is_training
            else False,
            seed=int(loader_config["seed"]),
        )

    return loaders
