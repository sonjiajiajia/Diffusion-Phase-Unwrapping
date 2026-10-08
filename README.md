# SNAPHU-Conditioned Diffusion for InSAR Phase Unwrapping

A conditional diffusion pipeline that predicts unwrapped InSAR phase using
SNAPHU unwrapping results as conditioning images. The repository includes
training, DDIM inference, overlapping-tile stitching, and mask-guided inference.

The model condition is **SNAPHU unwrapped phase**, not wrapped phase.
Wrapped phase may be provided for visualization only. The current pipeline
does not apply Itoh, SB-PU, or other physical sampling projections.

## Installation

Clone the repository and run these commands from its root, using a Python
environment with a suitable PyTorch installation:

```bash
python -m pip install -e .
```

Dependencies include TIFF processing, visualization, and mask processing.
MPI is optional:

```bash
python -m pip install -e '.[mpi]'
```

The Bash launchers use single-process training. Activate your Python
environment first, or set `PYTHON_BIN` to its interpreter. Dataset1.5 training
has been tested with Python 3.10 on an RTX 5090.

Datasets and pretrained weights are not included. Prepare SNAPHU conditions
externally; this repository does not run SNAPHU itself.

## Data Preparation

Training and validation splits must contain matching TIFF filenames:

```text
split/
  unwrapped/
    1.tif
    2.tif
  cond/
    1.tif
    2.tif
```

- `unwrapped`: ground-truth unwrapped phase in radians.
- `cond`: corresponding SNAPHU unwrapped phase in radians.
- Use finite, single-channel arrays with matching spatial dimensions.
- Default training uses 256 x 256 patches; the loader does not resize or crop.
- Training clips both arrays to `[-50, 50]` and maps them to `[-1, 1]`.
  Inference divides conditions by 50 without clipping. Use inputs compatible
  with the training range.

The Dataset1.5 launcher uses `DDPM_1w` for training and `DDPM_1k` for validation:

```text
Dataset1.5/
  DDPM_1w/
    unwrapped/
    cond/
  DDPM_1k/
    unwrapped/
    cond/
```

Its default dataset root is `$HOME/Data/Dataset1.5`. Set `DATASET_ROOT` to
use another location.

## Training

```bash
DATASET_ROOT=/path/to/Dataset1.5 bash run_dataset15_train.sh
```

Default settings for the Dataset1.5 training launcher:

| Setting | Value |
| --- | --- |
| Training / validation splits | DDPM_1w / DDPM_1k |
| Training patch size | 256 x 256 |
| Input / output channels | 2 / 1 |
| Base channels / residual blocks / attention heads | 128 / 2 / 4 |
| Diffusion steps / noise schedule | 1000 / linear |
| Prediction target | Normalized unwrapped phase (`predict_xstart=True`) |
| Learning rate | 1e-4, linearly annealed |
| Training updates | 150000 |
| Training batch / precision | 8 / FP32 |
| Maximum gradient norm | 1.0 (before the optimizer update) |
| EMA rate | 0.9999 |
| Validation interval / DDIM steps | 5000 / 100 |
| Validation batch / maximum batches per validation | 2 / 5 |
| Checkpoint interval | 5000 |

The two input channels are noisy target phase and the SNAPHU condition.
The default training objective is normalized unwrapped-phase prediction MSE:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
loss = mean((x0_theta(x_t, t, cond) - x_0)^2)
```

No rewrapping, phase-closure, gradient, or learned-variance loss is added by
default. An optional `loss_mask` restricts MSE to valid pixels without adding
a physical loss. `--predict_xstart False` selects the legacy noise-prediction
objective. Training and inference must use the same prediction mode. These
defaults describe the current code, not the verified training configuration
of an independently supplied checkpoint.

Override settings using environment variables or additional arguments:

```bash
DATASET_ROOT=/path/to/Dataset1.5 TRAIN_STEPS=20000 BATCH_SIZE=4 \
  USE_FP16=False OUT_DIR=runs/my_experiment bash run_dataset15_train.sh
```

For other datasets, use the generic launcher with an explicit validation interval:

```bash
DATA_DIR=/path/to/train VAL_DATA_DIR=/path/to/val OUT_DIR=runs/custom \
  bash run_train.sh --lr_anneal_steps 20000 --val_every 2000
```

The launcher uses `--microbatch -1` by default. You can split batches with
`--microbatch 2` if memory is limited; gradients are normalized by the full
batch size, including when the final microbatch is smaller.

### Resume Training

```bash
DATASET_ROOT=/path/to/Dataset1.5 \
  RESUME_CHECKPOINT=runs/dataset15_xstart/model010000.pt \
  TRAIN_STEPS=20000 bash run_dataset15_train.sh
```

Resume with a `model*.pt` checkpoint. Matching EMA and optimizer files are
loaded when available. `TRAIN_STEPS` is the total limit including resumed
steps, not the number of additional updates.

### Logs and Outputs

The default output directory is `runs/dataset15_xstart/`, configurable using
`OUT_DIR`. It contains training weights (`model*.pt`), EMA weights (`ema_*.pt`),
optimizer states (`opt*.pt`), `progress.csv`, `log.txt`, and validation figures
in `val_vis/`. `config.json` records the run settings, including prediction
mode. Resuming checks that mode when the checkpoint directory contains a
configuration file; legacy checkpoints without it cannot be checked automatically.

Console output and `log.txt` use the same compact format; full metrics remain
in `progress.csv`:

```text
step=100 | loss=0.178 | grad_norm=3.21 | lr=0.0001
```

FP32 is the default. `MAX_GRAD_NORM` controls gradient clipping; the logged
`grad_norm` is the norm before clipping. Set it to `0` to disable clipping.
Non-finite data, loss, or unscaled gradient norms stop training before an
optimizer update. Optional FP16 (`USE_FP16=True`) skips scaled-gradient
overflows, but stops if the loss scale reaches 1 without finite gradients.

## Inference

Supply a compatible checkpoint via `MODEL_PATH`. The default path is
`checkpoints/igarss_model.pt`; prefer EMA weights when available.

The x0-prediction run must start from scratch: do not resume its training from
a noise-prediction checkpoint. For inference with an existing noise-prediction
checkpoint, set `PREDICT_XSTART=False`. Weight shapes alone do not encode the
prediction objective. The new default output directory separates these runs.

EMA with rate 0.9999 retains approximately 67% of its initialization after
4000 successful updates. Early EMA validation can lag behind training weights;
do not interpret a small training loss as evidence of converged sampling.
At every successful optimizer update, EMA weights become `0.9999 * previous
EMA + 0.0001 * current weights`. Validation uses EMA weights, and each
checkpoint saves both `model*.pt` and `ema_*.pt` for the same step.

For 256 x 256 inputs, override the default tile dimensions:

```bash
MODEL_PATH=/path/to/ema_model.pt COND_DIR=/path/to/snaphu_cond \
  EXP_DIR=outputs/predictions \
  bash run_val.sh --tile_h 256 --tile_w 256 --overlap 64
```

For large images, `run_val.sh` defaults to 1024 x 1024 tiles with 128-pixel
overlap. Tile dimensions must not exceed the input and should be multiples
of the U-Net downsampling factor (32). Reduce tile size if memory is limited.

Inference uses 100 DDIM steps with `eta=0`, clips predicted normalized phase
to `[-1, 1]`, and multiplies it by 50. Tiles use mean offset alignment and
Hann-window blending. Initial sampling noise is random; the script currently
has no seed option.

Without ground truth, predictions are saved in `preds_tif/`. For evaluation:

```bash
MODEL_PATH=/path/to/ema_model.pt COND_DIR=/path/to/snaphu_cond \
  GT_DIR=/path/to/ground_truth EXP_DIR=outputs/evaluation \
  bash run_val.sh --tile_h 256 --tile_w 256 --overlap 64
```

With matching ground truth, outputs include `metrics.csv`, composite PNGs,
and TIFF predictions. Metrics are range-normalized RMSE (percent), global
SSIM, and MAE. By default, predictions are adjusted using the median
GT-minus-prediction offset before evaluation **and saving**. Disable that
adjustment with `--align_method none`. Comparing against the SNAPHU condition
itself does not measure independent ground-truth accuracy.

`WRAP_DIR` optionally supplies wrapped-phase TIFFs for figures, not model inputs.

## Mask-Guided Inference

```bash
MODEL_PATH=/path/to/ema_model.pt COND_DIR=/path/to/snaphu_cond \
  GT_DIR=/path/to/ground_truth MASK_MAT=/path/to/results.mat \
  MASK_KEY=Mask_Error EXP_DIR=outputs/masked \
  bash run_mask_val.sh --tile_h 256 --tile_w 256 --overlap 64
```

The script loads one shared MATLAB mask. Finite, nonzero values mark valid
pixels; zero or NaN values are invalid. It resizes the mask if needed and
applies hole filling and binary closing before selecting tiles.

Only tiles intersecting the processed mask are sampled. Outputs include
full tile-coverage predictions, strictly masked predictions, PNGs, and a CSV
of selected tile counts and alignment offsets. The mask is not an additional
model input channel and does not contribute to the training loss.

The current mask script requires matching ground truth for offset alignment
and skips images without it. Use `run_val.sh` for inference without ground truth.

## Repository Structure

```text
improved_diffusion/          Model, diffusion, data loading, training, logging
scripts/image_train.py      Training entry point
scripts/cond_sample_eval.py Conditional inference and evaluation
scripts/mask_tiling_eval.py Mask-guided tiled inference
run_train.sh               Generic training launcher
run_dataset15_train.sh     Dataset1.5 training launcher
run_val.sh                 Conditional inference launcher
run_mask_val.sh            Mask-guided inference launcher
setup.py                   Package and dependency declaration
THIRD_PARTY_NOTICES         Upstream copyright and license notices
```

Upstream notices for retained OpenAI code are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
