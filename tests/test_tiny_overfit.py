
from pathlib import Path

import pytest
import torch

from src.diffusion.schedule import (
    DDPMSchedule,
)
from src.training.tiny_overfit import (
    TinyOverfitTrainer,
)


class DummyTextEncoder(torch.nn.Module):

    pad_token_id = 1

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            32,
            8,
            padding_idx=1,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        hidden_states = self.embedding(
            input_ids
        )
        return hidden_states * attention_mask.unsqueeze(-1)


class DummyConditionalUNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Conv2d(
            3,
            3,
            kernel_size=1,
        )
        self.context_projection = torch.nn.Linear(
            8,
            3,
        )

    def forward(
        self,
        noisyimages,
        timesteps,
        context,
        attention_mask,
    ):
        pooled_context = (
            context.sum(dim=1)
            / attention_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=1)
        )

        bias = self.context_projection(
            pooled_context
        ).unsqueeze(-1).unsqueeze(-1)

        return self.convolution(
            noisyimages
        ) + bias


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA non disponibile",
)
def test_tiny_overfit_step_eval_and_checkpoint(
    tmp_path,
):
    device = torch.device("cuda")

    fixed_batch = {
        "image": torch.randn(
            4,
            3,
            32,
            32,
        ),
        "input_ids": torch.randint(
            0,
            16,
            (4, 6),
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(
            4,
            6,
            dtype=torch.long,
        ),
    }

    conditional_unet = (
        DummyConditionalUNet().to(device)
    )
    text_encoder = DummyTextEncoder().to(device)

    schedule = DDPMSchedule(
        num_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": conditional_unet.parameters(),
                "lr": 1e-3,
            },
            {
                "params": text_encoder.parameters(),
                "lr": 1e-3,
            },
        ]
    )

    config = {
        "overfit": {
            "seed": 2026,
            "gradient_clip_norm": 1.0,
            "amp": {
                "enabled": True,
                "dtype": "float16",
            },
            "evaluation": {
                "seed": 3107,
            },
        }
    }

    trainer = TinyOverfitTrainer(
        conditional_unet=conditional_unet,
        text_encoder=text_encoder,
        schedule=schedule,
        optimizer=optimizer,
        fixed_batch=fixed_batch,
        device=device,
        config=config,
    )

    metrics = trainer.train_step()

    assert metrics["step"] == 1
    assert torch.isfinite(
        torch.tensor(
            metrics["train_loss"],
            device=device,
        )
    )

    evaluation_loss = trainer.evaluate()

    assert torch.isfinite(
        torch.tensor(
            evaluation_loss,
            device=device,
        )
    )

    trainer.history.append(
        {
            **metrics,
            "evaluation_loss": evaluation_loss,
        }
    )

    checkpoint_path = (
        tmp_path / "tiny_overfit.pt"
    )

    trainer.save(checkpoint_path)

    assert checkpoint_path.exists()

    restored_unet = (
        DummyConditionalUNet().to(device)
    )
    restored_text_encoder = (
        DummyTextEncoder().to(device)
    )

    restored_schedule = DDPMSchedule(
        num_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
    ).to(device)

    restored_optimizer = torch.optim.AdamW(
        [
            {
                "params": restored_unet.parameters(),
                "lr": 1e-3,
            },
            {
                "params": restored_text_encoder.parameters(),
                "lr": 1e-3,
            },
        ]
    )

    restored_trainer = TinyOverfitTrainer(
        conditional_unet=restored_unet,
        text_encoder=restored_text_encoder,
        schedule=restored_schedule,
        optimizer=restored_optimizer,
        fixed_batch=fixed_batch,
        device=device,
        config=config,
    )

    restored_trainer.load(checkpoint_path)

    assert restored_trainer.step == trainer.step
    assert len(restored_trainer.history) == len(
        trainer.history
    )
