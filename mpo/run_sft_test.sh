#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
set -e

# GPU ưu tiên: 7 > 5
# GPU 7: free hoàn toàn
# GPU 5: còn ~62GB

echo "=== SFT model=4 (llama7b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 1 DPO 1 

