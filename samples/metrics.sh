#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../cpfusion/metrics.py"
FUSE_NAME="f1"
# FUSE_NAME="deepfuse"

python $PYTHON_SCRIPT \
    --dataset_path "${BASE_PATH}/torchvision"\
    --fuse_path "${BASE_PATH}/temp/${FUSE_NAME}"\
    --fuse_name "${FUSE_NAME}"\
    --metrics_path "${BASE_PATH}/temp/${FUSE_NAME}"

