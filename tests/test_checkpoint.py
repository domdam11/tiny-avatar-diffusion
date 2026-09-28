
import random

import numpy as np
import torch

from src.utils.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_checkpoint_restores_model_optimizer(
    tmp_path,
):
    torch.manual_seed(10)

    model = torch.nn.Linear(
        4,
        3,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    inputs = torch.randn(
        2,
        4,
    )

    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()

    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_training_checkpoint(
        path=checkpoint_path,
        models={"model": model},
        optimizer=optimizer,
        scaler=None,
        step=17,
        epoch=2,
        metadata={"best_loss": 0.5},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)

    checkpoint = (
        load_training_checkpoint(
            path=checkpoint_path,
            models={"model": model},
            optimizer=optimizer,
            map_location="cpu",
        )
    )

    assert checkpoint["step"] == 17
    assert checkpoint["epoch"] == 2

    for name, parameter in (
        model.named_parameters()
    ):
        torch.testing.assert_close(
            parameter,
            expected_parameters[name],
        )


def test_checkpoint_restores_rng(
    tmp_path,
):
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    model = torch.nn.Linear(
        2,
        2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters()
    )

    path = tmp_path / "rng.pt"

    save_training_checkpoint(
        path=path,
        models={"model": model},
        optimizer=optimizer,
        scaler=None,
        step=0,
    )

    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(4)

    for _ in range(10):
        random.random()
        np.random.rand()
        torch.rand(4)

    load_training_checkpoint(
        path=path,
        models={"model": model},
        optimizer=optimizer,
        map_location="cpu",
        restore_rng=True,
    )

    assert random.random() == (
        expected_python
    )

    assert np.random.rand() == (
        expected_numpy
    )

    torch.testing.assert_close(
        torch.rand(4),
        expected_torch,
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_restores_generator(
    tmp_path,
):
    model = torch.nn.Linear(
        2,
        2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters()
    )

    generator = (
        torch.Generator()
        .manual_seed(30)
    )

    path = tmp_path / "generator.pt"

    save_training_checkpoint(
        path=path,
        models={"model": model},
        optimizer=optimizer,
        scaler=None,
        step=5,
        generators={
            "training": generator
        },
    )

    expected = torch.randn(
        5,
        generator=generator,
    )

    torch.randn(
        20,
        generator=generator,
    )

    load_training_checkpoint(
        path=path,
        models={"model": model},
        optimizer=optimizer,
        generators={
            "training": generator
        },
        map_location="cpu",
    )

    actual = torch.randn(
        5,
        generator=generator,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_is_atomic(
    tmp_path,
):
    model = torch.nn.Linear(
        2,
        2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters()
    )

    path = tmp_path / "atomic.pt"

    save_training_checkpoint(
        path=path,
        models={"model": model},
        optimizer=optimizer,
        scaler=None,
        step=1,
    )

    assert path.exists()

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    assert not temporary.exists()
