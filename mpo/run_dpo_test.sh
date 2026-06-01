#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU ưu tiên: 7 > 5
# Thứ tự model: 3 (qwen3b) → 4 (llama7b) → 1 (qwen05b) → 2 (tinyllama11b)

# ================================================================
# Experiment 1: MPO-NoProj (variant 38)
# ================================================================
echo "=== MPO-NoProj model=3 (qwen3b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 1 MPO-NoProj 0 4
