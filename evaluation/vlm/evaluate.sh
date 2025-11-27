# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23459}
GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

DATASET=${1}
echo "CHECKPOINT: ${CHECKPOINT}"

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"

if  [ ${DATASET} == "mme" ]; then
  python -m vlm.eval.mme.eval "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmvet" ]; then
    python -m vlm.eval.mmvet.evaluate_mmvet --datasets mmvet "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_20230712 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_cn_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_en_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_cn_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_dev "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_validation "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-val_cot" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmmu.evaluate_mmmu_cot --datasets MMMU_validation_cot "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_test "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_testmini "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_test "${ARGS[@]:1}"
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m vlm.eval.pope.evaluate_pope --datasets pope "${ARGS[@]:1}"
fi

if [ ${DATASET} == "pope_cot" ]; then
    torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m vlm.eval.pope.evaluate_pope --datasets pope_cot --cot "${ARGS[@]:1}"
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m vlm.eval.vqa.evaluate_vqa --datasets gqa_testdev_llava "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=$WORLD_SIZE \
      --node_rank=$RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m vlm.eval.mmvp.evaluate_mmvp --datasets MMVP "${ARGS[@]:1}"
fi
