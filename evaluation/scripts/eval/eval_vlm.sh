# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Check if enough arguments are provided
# if [ $# -lt 2 ]; then
#     echo "Error: PREFIX_DIR and MODEL_PATH are required as the first and second arguments respectively."
#     exit 1
# fi

LOG_PATH=$1
FULL_MODEL_PATH=$2
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
shift 1
ARGS=("$@")
export MASTER_PORT=10042

IFS=' ' read -r -a DATASETS <<< "$DATASETS_STR"

for DATASET in "${DATASETS[@]}"; do
    bash vlm/evaluate.sh \
        "$DATASET" \
        --out-dir "$LOG_PATH/$DATASET" \
        --model-path "$FULL_MODEL_PATH" 
done