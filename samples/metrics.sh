#!/bin/bash

# Get Base Path

# cd "$(dirname "$0")"
# BASE_PATH=$("../../check_path.sh")
# if [ -z "$BASE_PATH" ]; then
#     echo "BASE_PATH Not Find"
#     exit 1
# fi


# Run Script

BASE_PATH="/Volumes/Charles/data/vision"
PYTHON_SCRIPT="./cpfusion/metrics.py"
FUSE_NAME="cpfusion_wp"
# FUSE_NAME="deepfuse"

python $PYTHON_SCRIPT \
    --dataset_path "${BASE_PATH}/torchvision"\
    --fuse_path "${BASE_PATH}/torchvision/tno/tno/fused/${FUSE_NAME}"\
    --fuse_name "${FUSE_NAME}"\
    --metrics_path "${BASE_PATH}/torchvision/tno/tno/fused"

