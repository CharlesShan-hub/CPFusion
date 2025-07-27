#!/bin/bash

PYTHON_SCRIPT="./cpfusion/inference.py"

python $PYTHON_SCRIPT \
    --ir_path "./data/ir/190015.jpg" \
    --vis_path "./data/vis/190015.jpg"
