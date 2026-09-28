
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 1


def _as_cpu_byte_tensor(
    state: torch.Tensor,
    state_name: str,
) -> torch.Tensor:
    """
    Converte uno stato RNG nel formato richiesto
    dalle API PyTorch: ByteTensor contiguo su CPU.
    """
    if not isinstance(state, torch.Tensor):
        raise TypeError(
            f"{state_name} deve essere un torch.Tensor, "
            f"ricevuto {type(state).__name__}."
        )

    return (
        state
        .detach()
        .to(
            device="cpu",
            dtype=torch.uint8,
        )
        .contiguous()
        .clone()
    )


def capture_rng_state() -> dict:
    """
    Cattura gli stati RNG globali di Python, NumPy,
    PyTorch CPU e, se disponibile, CUDA.
    """
    torch_cpu_state = _as_cpu_byte_tensor(
        torch.get_rng_state(),
        state_name="torch CPU RNG state",
    )

    torch_cuda_states = None

    if torch.cuda.is_available():
        torch_cuda_states = [
            _as_cpu_byte_tensor(
                state,
                state_name=(
                    f"torch CUDA RNG state {index}"
                ),
            )
            for index, state in enumerate(
                torch.cuda.get_rng_state_all()
            )
        ]

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_cpu_state,
        "torch_cuda": torch_cuda_states,
    }


def restore_rng_state(
    state: dict,
) -> None:
    """
    Ripristina gli stati RNG globali.

    Gli stati PyTorch vengono sempre convertiti in
    ByteTensor CPU prima del ripristino, anche se il
    checkpoint è stato caricato con map_location CUDA.
    """
    if not isinstance(state, dict):
        raise TypeError(
            "Lo stato RNG deve essere un dizionario."
        )

    required_keys = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }

    missing_keys = required_keys - set(state)

    if missing_keys:
        raise KeyError(
            "Chiavi RNG mancanti nel checkpoint: "
            f"{sorted(missing_keys)}"
        )

    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch_cpu_state = _as_cpu_byte_tensor(
        state["torch_cpu"],
        state_name="rng_state['torch_cpu']",
    )

    torch.set_rng_state(
        torch_cpu_state
    )

    cuda_states = state.get(
        "torch_cuda"
    )

    if (
        cuda_states is not None
        and torch.cuda.is_available()
    ):
        if not isinstance(
            cuda_states,
            (list, tuple),
        ):
            raise TypeError(
                "rng_state['torch_cuda'] deve essere "
                "una lista o una tupla."
            )

        restored_cuda_states = [
            _as_cpu_byte_tensor(
                cuda_state,
                state_name=(
                    "rng_state['torch_cuda']"
                    f"[{index}]"
                ),
            )
            for index, cuda_state in enumerate(
                cuda_states
            )
        ]

        available_device_count = (
            torch.cuda.device_count()
        )

        if (
            len(restored_cuda_states)
            != available_device_count
        ):
            raise RuntimeError(
                "Il checkpoint contiene stati RNG "
                f"per {len(restored_cuda_states)} GPU, "
                "ma il runtime corrente dispone di "
                f"{available_device_count} GPU."
            )

        torch.cuda.set_rng_state_all(
            restored_cuda_states
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
    """
    Salva atomicamente un checkpoint completo per
    la ripresa del training.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if step < 0:
        raise ValueError(
            "step non può essere negativo."
        )

    if epoch < 0:
        raise ValueError(
            "epoch non può essere negativa."
        )

    if not models:
        raise ValueError(
            "models non può essere vuoto."
        )

    model_states = {
        name: model.state_dict()
        for name, model in models.items()
    }

    generator_states = {}

    if generators is not None:
        generator_states = {
            name: _as_cpu_byte_tensor(
                generator.get_state(),
                state_name=(
                    f"generator_states[{name!r}]"
                ),
            )
            for name, generator
            in generators.items()
        }

    checkpoint = {
        "format_version": (
            CHECKPOINT_FORMAT_VERSION
        ),
        "step": int(step),
        "epoch": int(epoch),
        "models": model_states,
        "optimizer": optimizer.state_dict(),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "rng_state": capture_rng_state(),
        "generator_states": generator_states,
        "metadata": (
            metadata
            if metadata is not None
            else {}
        ),
        "configs": (
            configs
            if configs is not None
            else {}
        ),
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        torch.save(
            checkpoint,
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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
    """
    Carica un checkpoint completo e ripristina
    modelli, optimizer, scaler e stati RNG.
    """
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

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Il checkpoint deve essere "
            "un dizionario."
        )

    format_version = checkpoint.get(
        "format_version"
    )

    if (
        format_version
        != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "Versione checkpoint non supportata: "
            f"{format_version}. Versione attesa: "
            f"{CHECKPOINT_FORMAT_VERSION}."
        )

    required_keys = {
        "step",
        "epoch",
        "models",
        "optimizer",
        "scaler",
        "rng_state",
        "generator_states",
        "metadata",
        "configs",
    }

    missing_keys = (
        required_keys
        - set(checkpoint)
    )

    if missing_keys:
        raise KeyError(
            "Chiavi mancanti nel checkpoint: "
            f"{sorted(missing_keys)}"
        )

    saved_models = checkpoint["models"]

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
        optimizer_state = checkpoint.get(
            "optimizer"
        )

        if optimizer_state is None:
            raise ValueError(
                "È stato richiesto il ripristino "
                "dell'optimizer, ma il checkpoint "
                "non contiene il relativo stato."
            )

        optimizer.load_state_dict(
            optimizer_state
        )

    if scaler is not None:
        scaler_state = checkpoint.get(
            "scaler"
        )

        if scaler_state is not None:
            scaler.load_state_dict(
                scaler_state
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
                "Generatori del checkpoint diversi "
                "da quelli richiesti. "
                f"Salvati="
                f"{set(saved_generator_states)}, "
                f"richiesti={set(generators)}."
            )

        for name, generator in (
            generators.items()
        ):
            generator_state = (
                _as_cpu_byte_tensor(
                    saved_generator_states[
                        name
                    ],
                    state_name=(
                        "generator_states"
                        f"[{name!r}]"
                    ),
                )
            )

            generator.set_state(
                generator_state
            )

    if restore_rng:
        restore_rng_state(
            checkpoint["rng_state"]
        )

    return checkpoint
