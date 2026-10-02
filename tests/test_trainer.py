
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.diffusion.schedule import DDPMSchedule
from src.training.trainer import Trainer

class DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
    def forward(self, noisy_images, timesteps, context, attention_mask):
        return self.conv(noisy_images)

class DummyTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 16)
    def forward(self, input_ids, attention_mask):
        return self.emb(input_ids)

@pytest.fixture
def dummy_setup():
    unet = DummyUNet()
    text_encoder = DummyTextEncoder()
    schedule = DDPMSchedule(num_timesteps=10)
    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()))
    
    # Dummy data
    images = torch.randn(8, 3, 32, 32)
    input_ids = torch.randint(0, 10, (8, 5))
    attention_mask = torch.ones(8, 5)
    
    dataset = [{"image": images[i], "input_ids": input_ids[i], "attention_mask": attention_mask[i]} for i in range(8)]
    loader = DataLoader(dataset, batch_size=4)
    
    config = {
        "training": {
            "gradient_clip_norm": 1.0,
            "amp": {"enabled": False},
            "seed": 42
        }
    }
    
    return unet, text_encoder, schedule, optimizer, loader, config

def test_trainer_initialization_and_step(dummy_setup):
    unet, text_encoder, schedule, optimizer, loader, config = dummy_setup
    device = torch.device("cpu")
    
    trainer = Trainer(
        conditional_unet=unet,
        text_encoder=text_encoder,
        schedule=schedule,
        optimizer=optimizer,
        train_dataloader=loader,
        device=device,
        config=config,
    )
    
    batch = next(iter(loader))
    metrics = trainer.train_step(batch)
    
    assert trainer.step == 1
    assert "train_loss" in metrics
    assert "gradient_norm" in metrics
