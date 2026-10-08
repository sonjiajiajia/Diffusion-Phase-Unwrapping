#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-checkpoints/igarss_model.pt}"
COND_DIR="${COND_DIR:-data/test/cond}"
GT_DIR="${GT_DIR:-data/test/unwrapped}"
MASK_MAT="${MASK_MAT:-data/test/results.mat}"
MASK_KEY="${MASK_KEY:-Mask_Error}"
EXP_DIR="${EXP_DIR:-outputs/igarss_mask_results}"

"${PYTHON_BIN}" scripts/mask_tiling_eval.py \
  --model_path "${MODEL_PATH}" \
  --cond_dir "${COND_DIR}" \
  --gt_dir "${GT_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --image_size 256 \
  --use_ddim True  \
  --timestep_respacing 100 \
  --denorm 50 \
  --predict_xstart "${PREDICT_XSTART:-True}" \
  --align_method median \
  --tile_bias_align mean \
  --save_tif True \
  --in_channels 2 --out_channels 1 \
  --num_channels 128 --num_res_blocks 2 --num_heads 4 \
  --clip_denoised True \
  --tile_h 512 --tile_w 512 --overlap 128 \
  --mask_mat "${MASK_MAT}" \
  --mask_key "${MASK_KEY}" \
  "$@"
