#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/train}"
VAL_DATA_DIR="${VAL_DATA_DIR:-data/val}"
OUT_DIR="${OUT_DIR:-runs/snaphu_train}"
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENAI_USE_DDP=0
unset RANK WORLD_SIZE LOCAL_RANK
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" scripts/image_train.py \
  --out_dir "${OUT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --val_data_dir "${VAL_DATA_DIR}" \
  --image_size 256 --in_channels 2 --out_channels 1 \
  --num_channels 128 --num_res_blocks 2 --num_heads 4 \
  --diffusion_steps 1000 --noise_schedule linear \
  --predict_xstart "${PREDICT_XSTART:-True}" \
  --batch_size 8 --val_batch_size 8 \
  --lr "${LR:-1e-4}" --use_fp16 "${USE_FP16:-False}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --val_use_ddim True --val_timestep_respacing 100 \
  --log_interval 1000 --save_interval 10000 --val_every 5000 \
  --val_max_batches 10 \
  "${RESUME_ARGS[@]}" \
  "$@"
