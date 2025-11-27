# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export GPUS=8

output_path="your_save_path"
model_path='your_ckpt_path'
DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mmvp" "pope")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    $model_path