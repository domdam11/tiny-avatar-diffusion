
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: dict,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(
        state["torch_cpu"]
    )

    cuda_state = state.get(
        "torch_cuda"
    )

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_state
        )


def save_training_checkpoint(
    path: Path,
    models: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[
        torch.amp.GradScaler
    ],
    step: int,
    epoch: int = 0,
    metadata: Optional[dict] = None,
    configs: Optional[dict] = None,
    generators: Optional[
        dict[str, torch.Generator]
    ] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if step < 0:
        raise ValueError(
            "step non può essere negativo."
        )

    model_states = {
        name: model.state_dict()
        for name, model in models.items()
    }

    generator_states = {}

    if generators is not None:
        generator_states = {
            name: generator.get_state()
            for name, generator
            in generators.items()
        }

    checkpoint = {
        "format_version": 1,
        "step": int(step),
        "epoch": int(epoch),
        "models": model_states,
        "optimizer": (
            optimizer.state_dict()
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "rng_state": capture_rng_state(),
        "generator_states": (
            generator_states
        ),
        "metadata": metadata or {},
        "configs": configs or {},
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def load_training_checkpoint(
    path: Path,
    models: dict[str, nn.Module],
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    scaler: Optional[
        torch.amp.GradScaler
    ] = None,
    generators: Optional[
        dict[str, torch.Generator]
    ] = None,
    map_location: Any = "cpu",
    restore_rng: bool = True,
) -> dict:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if checkpoint.get(
        "format_version"
    ) != 1:
        raise ValueError(
            "Versione checkpoint non supportata."
        )

    saved_models = checkpoint[
        "models"
    ]

    if set(saved_models) != set(models):
        raise ValueError(
            "Modelli nel checkpoint diversi "
            "da quelli richiesti. "
            f"Salvati={set(saved_models)}, "
            f"richiesti={set(models)}."
        )

    for name, model in models.items():
        model.load_state_dict(
            saved_models[name],
            strict=True,
        )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if (
        scaler is not None
        and checkpoint["scaler"]
        is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    if generators is not None:
        saved_generator_states = (
            checkpoint.get(
                "generator_states",
                {},
            )
        )

        if (
            set(saved_generator_states)
            != set(generators)
        ):
            raise ValueError(
                "Generatori del checkpoint "
                "diversi da quelli richiesti."
            )

        for name, generator in (
            generators.items()
        ):
            generator.set_state(
                saved_generator_states[
                    name
                ]
            )

    if restore_rng:
        restore_rng_state(
            checkpoint["rng_state"]
        )

    return checkpoint
