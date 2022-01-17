#!/bin/bash

# -------------------------------------------------------
# Please update the following part.
# -------------------------------------------------------
MODEL_NAME=qnapaicore_v1-6
TYPE=onnx
MODEL_FILE=/home/rknn/data/$MODEL_NAME.$TYPE
# -------------------------------------------------------

rm -rf ./$MODEL_NAME.*
rm -rf ./pytorch_$MODEL_NAME.npy

if [[ -f $MODEL_FILE ]]; then
  echo "Got the latest PyTorch model."
  cp $MODEL_FILE ./
  python3 test.py ./$MODEL_NAME.$TYPE
  if [[ $? -eq 0 ]]; then
    echo "The RKNN model generated on the current directory."
  fi
else
  echo "The model $MODEL_FILE can't be found."
fi

# rm -rf ./$MODEL_NAME.rknn
