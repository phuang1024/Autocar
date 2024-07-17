#!/bin/bash

python export.py $1 $1.onnx
python -m blobconverter --onnx $1.onnx --output $2 --shaves 6
