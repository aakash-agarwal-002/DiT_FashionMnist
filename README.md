# DiT_FashionMnist

**DiT_FashionMnist**: a collection of experiments and reference code for training Diffusion Transformer (DiT) models on the FashionMNIST dataset. This repository includes multiple model variants, training/inference scripts, checkpoints, and utilities used to run experiments, reproduce results, and generate sample images.

**Repository Overview**
- **`adaptive_layer_norm/`**: DiT variant using adaptive layer normalization. Contains training code, checkpoints, logs, and sample outputs.
- **`adaptive_layer_norm_zero/`**: Variant with zero-initialized adaptive layer normalization.
- **`in-context_conditioning/`**: Experiments that include in-context conditioning mechanisms.
- **`inference_optimization_ddim/`**: Inference optimizations and DDIM sampling experiments.
- **`no_conditioning/`**: Model variant trained without conditioning information.
- **`data/`**: Local datasets; includes `FashionMNIST` and `MNIST` raw files used by experiments.

**Quick Status**
- Several checkpoints are included in each variant folder under `checkpoints/` (e.g. `dit_epoch_25.pt`).
- Sample outputs, logs, and FID/loss tracking live under each variant's `results/` and `dit_log.txt`.

**Requirements**
- The project uses PyTorch and common machine-learning utilities. The `requirements.txt` contains the canonical list. Key packages:
	- `torch`
	- `torchvision`
	- `numpy`
	- `matplotlib`
	- `tqdm`
	- `torchmetrics`

**Setup**
Create and activate a Python environment (conda recommended):

```zsh
conda create -n dit_fmnist python=3.10 -y
conda activate dit_fmnist
pip install -r requirements.txt
```

**Preparing the Data**
- The repository already contains raw FashionMNIST files under `data/FashionMNIST/raw/` (these are the standard IDX files). If you prefer to download automatically via `torchvision`, you can run:

```zsh
python - <<'PY'
from torchvision.datasets import FashionMNIST
FashionMNIST(root='data/FashionMNIST', train=True, download=True)
FashionMNIST(root='data/FashionMNIST', train=False, download=True)
print('Downloaded FashionMNIST to data/FashionMNIST')
PY
```

If you already have IDX files, keep them in `data/FashionMNIST/raw/` and the training scripts will use them directly.

**How the Code Is Organized**
- Each experimental variant folder contains:
	- `dit.py`: main training / entrypoint script for that variant.
	- `vit_model.py`: ViT-based model definitions used by DiT.
	- `checkpoints/`: saved model checkpoints (e.g., `dit_epoch_25.pt`).
	- `results/`: generated images, loss logs, FID scores, and test outputs.

**Running Training**
- Basic usage (run from inside a variant directory):

```zsh
cd adaptive_layer_norm
python dit.py
```

- Notes:
	- Many hyperparameters are defined at the top of each `dit.py` file or in simple config sections. Edit those values to change learning rate, number of epochs, batch size, model size, and dataset paths.
	- Checkpoints are written to the local `checkpoints/` directory. Logs (loss, FID) are saved inside `results/` and `dit_log.txt`.

**Running Inference / Sampling**
- Example (using a saved checkpoint):

```zsh
cd adaptive_layer_norm
python dit.py # uses the last saved checkpoint
```

**Files of Interest**
- `checkpoints/dit_epoch_25.pt`: example saved model weights.
- `results/fid_scores.txt`: tracked FID scores across saved checkpoints.
- `results/loss.txt` and `dit_log.txt`: training loss and logging output.

**Reproducing the Provided Results**
1. Install dependencies (see Setup).
2. Ensure `data/FashionMNIST` exists (download via torchvision or use the included raw files).
3. Choose a variant folder (for the canonical DiT variant, use `adaptive_layer_norm/`).
4. Start training by running `python dit.py` inside that folder. Monitor `results/` and `dit_log.txt` for progress.
5. Use saved checkpoints in `checkpoints/` to run sampling/evaluation.

**Tips & Troubleshooting**
- GPU memory: reduce `batch_size` or `model_size` in the script if you run out of memory.
- Missing dependencies: re-run `pip install -r requirements.txt` and check for system-level dependencies required by `torch`.
- If `dit.py` lacks CLI parsing: open the script and edit configuration variables directly at the top of the file.

