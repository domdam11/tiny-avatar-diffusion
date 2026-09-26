# Tiny Text-Conditioned Avatar Diffusion

University project for the Deep Learning course at Politecnico di Bari.

## Objective

Implement and train from scratch a compact text-conditioned pixel-space
DDPM that generates cartoon avatars from deterministic attribute captions.

## Main components

- Reproducible preprocessing pipeline
- Compositional train/validation/test/OOD split
- Tokenizer trained from the training split
- Transformer text encoder trained from scratch
- Compact time-conditioned U-Net
- Unconditional baseline
- Text-conditioned diffusion model
- Quality, conditioning, diversity and efficiency evaluation

## Constraints

The main model does not use:

- Pretrained diffusion checkpoints
- Stable Diffusion
- Pretrained text encoders
- CLIP, BERT or T5 embeddings
- Pretrained VAEs
- Diffusers training pipelines as black boxes

## Initial configuration

- Dataset: Google Cartoon Set 10k
- Image resolution: 32x32 RGB
- Model: pixel-space DDPM
- Execution environment: Google Colab Pro
