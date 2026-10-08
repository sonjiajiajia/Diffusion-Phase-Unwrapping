# SNAPHU-Conditioned Diffusion for InSAR Phase Unwrapping

This repository contains the code accompanying the paper
[**An InSAR Phase Unwrapping Framework for Large-scale and Complex Events**](https://arxiv.org/abs/2603.21378).

The paper was nominated for the **IGARSS 2026 Student Paper Contest**.

A conditional diffusion pipeline that predicts unwrapped InSAR phase using
SNAPHU unwrapping results as conditioning images.  

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
environment first, or set `PYTHON_BIN` to its interpreter.

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

An example project data layout is:

```text
data/
  train/
    unwrapped/
    cond/
  val/
    unwrapped/
    cond/
  test/
    cond/
    unwrapped/   # Optional ground truth for evaluation
```

Use `DATA_DIR` and `VAL_DATA_DIR` to select the training and validation
splits. Use `COND_DIR` and optional `GT_DIR` for inference and evaluation.

## Training

```bash
DATA_DIR=/path/to/train VAL_DATA_DIR=/path/to/val OUT_DIR=runs/snaphu_train \
  bash run_train.sh --lr_anneal_steps 20000
```

Default settings for `run_train.sh`:

| Setting | Value |
| --- | --- |
| Training / validation directories | data/train / data/val |
| Training patch size | 256 x 256 |
| Input / output channels | 2 / 1 |
| Base channels / residual blocks / attention heads | 128 / 2 / 4 |
| Diffusion steps / noise schedule | 1000 / linear |
| Prediction target | Normalized unwrapped phase (`predict_xstart=True`) |
| Learning rate | 1e-4 |
| Training limit | Set with --lr_anneal_steps; unlimited when 0 |
| Training batch / precision | 8 / FP32 |
| Maximum gradient norm | 1.0 (before the optimizer update) |
| EMA rate | 0.9999 |
| Validation interval / DDIM steps | 5000 / 100 |
| Validation batch / maximum batches per validation | 8 / 10 |
| Checkpoint interval | 10000 |
| Logging interval | 1000 |

The command above runs to a total limit of 20000 steps and linearly anneals
the learning rate over that period. Change the limit for your experiment.
Validation samples up to 80 images per pass with the default settings.

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
DATA_DIR=/path/to/train VAL_DATA_DIR=/path/to/val OUT_DIR=runs/my_experiment \
  LR=5e-5 USE_FP16=False bash run_train.sh \
  --lr_anneal_steps 20000 --batch_size 4 --val_every 2000
```

The launcher uses `--microbatch -1` by default. You can split batches with
`--microbatch 2` if memory is limited; gradients are normalized by the full
batch size, including when the final microbatch is smaller.

### Resume Training

```bash
DATA_DIR=/path/to/train VAL_DATA_DIR=/path/to/val OUT_DIR=runs/snaphu_train \
  RESUME_CHECKPOINT=runs/snaphu_train/model010000.pt \
  bash run_train.sh --lr_anneal_steps 20000
```

Resume with a `model*.pt` checkpoint. Matching EMA and optimizer files are
loaded when available. `--lr_anneal_steps` is the total limit including resumed
steps, not the number of additional updates.

### Logs and Outputs

The default output directory is `runs/snaphu_train/`, configurable using
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

Training validation computes per-image `RMSE(pred - GT) / (GT.max - GT.min)`
and reports its mean as a ratio, without phase-offset alignment. Validation
figures show the original cond, prediction, and GT phases in radians.

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
prediction objective.

EMA smooths model weights across optimizer updates. At every successful
update, EMA weights become `0.9999 * previous EMA + 0.0001 * current weights`.
Validation uses EMA weights, which can lag behind training weights early in
training. Each checkpoint saves both `model*.pt` and `ema_*.pt`.

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
run_val.sh                 Conditional inference launcher
run_mask_val.sh            Mask-guided inference launcher
setup.py                   Package and dependency declaration
THIRD_PARTY_NOTICES         Upstream copyright and license notices
```

Upstream notices for retained OpenAI code are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
