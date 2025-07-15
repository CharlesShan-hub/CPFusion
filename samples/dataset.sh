#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../cpfusion/dataset.py"

python $PYTHON_SCRIPT \
    --dataset_path "${BASE_PATH}/torchvision" \
    --train_size 0.6 \
    --test_size 0.2 \
    --val_size 0.2 \
    --llvip_val_size 0.2 \
    --seed 42

