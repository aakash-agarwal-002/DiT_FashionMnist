# DiT_FashionMnist

This repository provides experiments and reference code for training Diffusion Transformer (DiT) models on the FashionMNIST dataset. It contains multiple model variants, training and inference scripts, checkpoints, and utilities for reproducing results and generating sample outputs.

## Repository Overview

- `adaptive_layer_norm/`  
  DiT variant using adaptive layer normalization. Includes training scripts, checkpoints, logs, and sample outputs.

- `adaptive_layer_norm_zero/`  
  Variant with zero initialized adaptive layer normalization.

- `in-context_conditioning/`  
  Experiments that incorporate in-context conditioning mechanisms.

- `inference_optimization_ddim/`  
  Inference optimizations and DDIM based sampling experiments.

- `no_conditioning/`  
  DiT trained without conditioning information.

- `data/`  
  Local datasets. Contains FashionMNIST and MNIST raw IDX files used by the experiments.

Every variant directory contains its own `dit.py`, `vit_model.py`, checkpoints, and results folder.

## Quick Status

- Checkpoints are located under `checkpoints/` in each variant folder, for example `dit_epoch_25.pt`.
- Sample outputs, logs, training loss, and FID measurements appear in each variant’s `results/` directory and in `dit_log.txt`.

## Requirements

Install Python dependencies from `requirements.txt`. Key packages include:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm`
- `torchmetrics`

## Setup

Create and activate an environment:

```zsh
conda create -n dit_fmnist python=3.10 -y
conda activate dit_fmnist
pip install -r requirements.txt
```

## Preparing the Data

The repository includes FashionMNIST raw IDX files under `data/FashionMNIST/raw/`.  
If you prefer an automatic download via torchvision, run:

```zsh
python - <<'PY'
from torchvision.datasets import FashionMNIST
FashionMNIST(root='data/FashionMNIST', train=True, download=True)
FashionMNIST(root='data/FashionMNIST', train=False, download=True)
print('Downloaded FashionMNIST to data/FashionMNIST')
PY
```

If you already have IDX files, leave them in `data/FashionMNIST/raw/`. Training scripts will load them automatically.

## How the Code Is Organized

Each experimental variant folder contains:

- `dit.py`  
  Main training and entrypoint script.
- `vit_model.py`  
  Vision Transformer based model definitions used in DiT.
- `checkpoints/`  
  Saved model weights.
- `results/`  
  Generated images, loss logs, FID scores, and test outputs.

## Parameters and Hyperparameters

| Parameter   | Value |
|-------------|-------|
| timesteps   | 1000  |
| emb_dim     | 256   |
| num_block   | 6     |
| heads       | 8     |
| ff_dim      | 4     |
| epochs      | 25    |
| patch_size  | 7     |
| lr          | 1e-3  |

These values govern the diffusion process, architectural depth and width, and all principal training settings.

## Dataset Note

The training code works out of the box with any image dataset that provides tensors of shape  
\((C, H, W)\) for each sample.  
The model automatically adapts to the input channel count and spatial resolution.  
To switch datasets, simply point your data loader to a new dataset following the same tensor shape convention.

## Running Training

From inside a variant directory:

```zsh
cd adaptive_layer_norm
python dit.py
```

Notes:

- Hyperparameters are defined at the top of `dit.py`. Modify them to adjust learning rate, number of epochs, batch size, model size, dataset paths, or architectural details.
- Checkpoints are written to `checkpoints/`.
- Logs and metrics appear in `results/` and `dit_log.txt`.

## Running Inference and Sampling

To generate samples using the latest checkpoint:

```zsh
cd adaptive_layer_norm
python dit.py
```

The script automatically loads the most recent checkpoint unless configured otherwise.

## Files of Interest

- `checkpoints/dit_epoch_25.pt`  
  Example model checkpoint.

- `results/fid_scores.txt`  
  Recorded FID values per saved checkpoint.

- `results/loss.txt` and `dit_log.txt`  
  Training loss and general logging output.

## Reproducing the Provided Results

1. Install dependencies.  
2. Ensure `data/FashionMNIST` exists via download or local raw IDX files.  
3. Navigate to a variant folder, for example `adaptive_layer_norm/`.  
4. Run `python dit.py` and monitor `results/` and `dit_log.txt`.  
5. Use the saved checkpoints for sampling or evaluation.

## Troubleshooting

- If GPU memory is insufficient, reduce `batch_size` or the model size.  
- If dependencies are missing, reinstall using `pip install -r requirements.txt`.  
- If no command line parser is provided in `dit.py`, modify hyperparameters directly inside the script.
