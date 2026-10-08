#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-checkpoints/igarss_model.pt}"
COND_DIR="${COND_DIR:-data/test/cond}"
GT_DIR="${GT_DIR:-}"
WRAP_DIR="${WRAP_DIR:-}"
EXP_DIR="${EXP_DIR:-outputs/igarss_results}"

"${PYTHON_BIN}" scripts/cond_sample_eval.py \
  --model_path "${MODEL_PATH}" \
  --cond_dir "${COND_DIR}" \
  --gt_dir "${GT_DIR}" \
  --wrap_dir "${WRAP_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --batch_size 8 \
  --image_size 256 \
  --use_ddim True \
  --timestep_respacing 100 \
  --denorm 50 \
  --predict_xstart "${PREDICT_XSTART:-True}" \
  --align_method median \
  --tile_bias_align mean \
  --save_tif True \
  --in_channels 2 \
  --out_channels 1 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --num_heads 4 \
  --clip_denoised True \
  --use_tiling True \
  --tile_h 1024 \
  --tile_w 1024 \
  --overlap 128 \
  "$@"
