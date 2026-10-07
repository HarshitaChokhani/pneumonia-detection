#!/usr/bin/env bash
# Full pipeline: train both models -> evaluate -> Grad-CAM
set -e
python -m src.train
python -m src.evaluate
python -m src.gradcam --model resnet50 --n 8
python -m src.gradcam --model baseline_cnn --n 8
echo "All outputs are in outputs/ (checkpoints, figures, metrics)"
