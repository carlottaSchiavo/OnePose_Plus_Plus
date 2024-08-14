#The SFM must be perform before using this script ! 

#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME=$1
echo "Current work dir: $PROJECT_DIR"

echo "-----------------------------------"
echo "Run inference and output demo video:"
echo "-----------------------------------"

# Run inference on $OBJ_NAME-test and output demo video:
python $PROJECT_DIR/demo.py +experiment="inference_demo" data_base_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-test" sfm_base_dir="$PROJECT_DIR/data/demo/sfm_model/outputs_softmax_loftr_loftr/$OBJ_NAME"