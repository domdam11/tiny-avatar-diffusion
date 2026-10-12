from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.diffusion.schedule import DDPMSchedule
from src.training.trainer import Trainer


class DummyUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(
        self,
        noisy_images,
        timesteps,
        context,
        attention_mask,
    ):
        del timesteps, context, attention_mask
        return self.conv(noisy_images)


class DummyTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 16)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.embedding(input_ids)


def build_loader() -> DataLoader:
    images = torch.randn(8, 3, 32, 32)
    input_ids = torch.randint(0, 10, (8, 5))
    attention_mask = torch.ones(8, 5, dtype=torch.long)

    dataset = [
        {
            "image": images[index],
            "input_ids": input_ids[index],
            "attention_mask": attention_mask[index],
        }
        for index in range(8)
    ]

    return DataLoader(dataset, batch_size=4, shuffle=False)


def build_config() -> dict:
    return {
        "training": {
            "seed": 42,
            "gradient_clip_norm": 1.0,
            "amp": {"enabled": False},
            "evaluation": {"seed": 3107},
        }
    }


def test_conditional_trainer_runs_one_step() -> None:
    unet = DummyUNet()
    text_encoder = DummyTextEncoder()
    schedule = DDPMSchedule(num_timesteps=10)
    loader = build_loader()

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(text_encoder.parameters())
    )

    trainer = Trainer(
        conditional_unet=unet,
        text_encoder=text_encoder,
        schedule=schedule,
        optimizer=optimizer,
        train_dataloader=loader,
        validation_dataloader=loader,
        device=torch.device("cpu"),
        config=build_config(),
    )

    metrics = trainer.train_step(next(iter(loader)))

    assert trainer.is_conditional
    assert trainer.step == 1
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
    assert torch.isfinite(torch.tensor(metrics["gradient_norm"]))
    assert torch.isfinite(torch.tensor(trainer.evaluate()))


def test_unconditional_trainer_runs_one_step() -> None:
    unet = DummyUNet()
    schedule = DDPMSchedule(num_timesteps=10)
    loader = build_loader()

    optimizer = torch.optim.AdamW(unet.parameters())

    trainer = Trainer(
        conditional_unet=unet,
        text_encoder=None,
        schedule=schedule,
        optimizer=optimizer,
        train_dataloader=loader,
        validation_dataloader=loader,
        device=torch.device("cpu"),
        config=build_config(),
    )

    metrics = trainer.train_step(next(iter(loader)))

    assert not trainer.is_conditional
    assert trainer.step == 1
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
    assert torch.isfinite(torch.tensor(metrics["gradient_norm"]))
    assert torch.isfinite(torch.tensor(trainer.evaluate()))


def test_trainer_checkpoint_resume(tmp_path: Path) -> None:
    loader = build_loader()
    config = build_config()
    schedule = DDPMSchedule(num_timesteps=10)

    unet = DummyUNet()
    text_encoder = DummyTextEncoder()

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(text_encoder.parameters())
    )

    trainer = Trainer(
        conditional_unet=unet,
        text_encoder=text_encoder,
        schedule=schedule,
        optimizer=optimizer,
        train_dataloader=loader,
        validation_dataloader=loader,
        device=torch.device("cpu"),
        config=config,
    )

    trainer.train_step(next(iter(loader)))
    trainer.best_evaluation_loss = trainer.evaluate()

    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save(checkpoint_path)

    resumed_unet = DummyUNet()
    resumed_text_encoder = DummyTextEncoder()

    resumed_optimizer = torch.optim.AdamW(
        list(resumed_unet.parameters())
        + list(resumed_text_encoder.parameters())
    )

    resumed_trainer = Trainer(
        conditional_unet=resumed_unet,
        text_encoder=resumed_text_encoder,
        schedule=DDPMSchedule(num_timesteps=10),
        optimizer=resumed_optimizer,
        train_dataloader=loader,
        validation_dataloader=loader,
        device=torch.device("cpu"),
        config=config,
    )

    resumed_trainer.load(checkpoint_path)

    assert resumed_trainer.step == trainer.step
    assert resumed_trainer.best_evaluation_loss == trainer.best_evaluation_loss
    assert len(resumed_optimizer.state) > 0
