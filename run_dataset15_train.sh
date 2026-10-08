#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/Data/Dataset1.5}"
export DATA_DIR="${DATA_DIR:-${DATASET_ROOT}/DDPM_1w}"
export VAL_DATA_DIR="${VAL_DATA_DIR:-${DATASET_ROOT}/DDPM_1k}"
export OUT_DIR="${OUT_DIR:-runs/dataset15_xstart}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

for split_dir in "${DATA_DIR}" "${VAL_DATA_DIR}"; do
  for subdir in unwrapped cond; do
    if [[ ! -d "${split_dir}/${subdir}" ]]; then
      printf 'Missing dataset directory: %s\n' "${split_dir}/${subdir}" >&2
      exit 1
    fi
  done
done

# Keep the full batch together by default.
bash "${PROJECT_DIR}/run_train.sh" \
  --batch_size "${BATCH_SIZE:-8}" --microbatch -1 \
  --val_batch_size "${VAL_BATCH_SIZE:-2}" \
  --use_fp16 "${USE_FP16:-False}" \
  --lr_anneal_steps "${TRAIN_STEPS:-150000}" \
  --log_interval "${LOG_INTERVAL:-200}" \
  --save_interval "${SAVE_INTERVAL:-5000}" \
  --val_every "${VAL_EVERY:-5000}" \
  --val_timestep_respacing 100 \
  --val_max_batches "${VAL_MAX_BATCHES:-5}" \
  --val_max_samples 0 --val_num_vis "${VAL_NUM_VIS:-4}" \
  "$@"
