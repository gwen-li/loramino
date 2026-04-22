# loramino

## Set up

```bash
conda create -n loramino python=3.10
conda activate loramino
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config_file config.json --base_model pythia-14m --verbose
```

### Run tiny test on local machine

```bash
python main.py --config_file config_tiny.json --base_model pythia-14m --verbose
```

## Compile cuda kernels

```bash
module load cuda
module load gcc
python3 - <<'PY'
import os
import torch
from loramino.adapters.cuda_extension import load_grouped_lora_cuda_extension, grouped_lora_cuda_error

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CC:", os.environ.get("CC"))
print("CXX:", os.environ.get("CXX"))
print("CUDAHOSTCXX:", os.environ.get("CUDAHOSTCXX"))

ext = load_grouped_lora_cuda_extension()
print("extension_loaded:", ext is not None)
print("extension_error:", grouped_lora_cuda_error())
PY
```