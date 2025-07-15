#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../cpfusion/msd.py"

python $PYTHON_SCRIPT \
    --dataset_path "${BASE_PATH}/torchvision"
