#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt

python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA disponibile:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA non disponibile: selezionare un runtime GPU in Colab."
    )
PY
