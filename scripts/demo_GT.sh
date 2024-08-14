#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME=$1
echo "Current work dir: $PROJECT_DIR"

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated & test sequence:
python $PROJECT_DIR/parse_scanned_data_GT.py \
    --scanned_object_path \
    "$PROJECT_DIR/data/demo/$OBJ_NAME"

