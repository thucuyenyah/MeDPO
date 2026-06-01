#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU ưu tiên: 7 > 5
# Thứ tự model: 3 (qwen3b) → 4 (llama7b) → 1 (qwen05b) → 2 (tinyllama11b)

# ================================================================
# Experiment 1: MPO-NoProj (variant 38)
# ================================================================
echo "=== MPO-NoProj model=3 (qwen3b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 1 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 2 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 3 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 4 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 5 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 6 MPO-NoProj 0 &
wait

echo "=== MPO-NoProj model=4 (llama7b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 1 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 2 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 3 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 4 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 5 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 6 MPO-NoProj 0 &
wait

echo "=== MPO-NoProj model=1 (qwen05b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 1 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 2 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 3 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 4 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 5 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 6 MPO-NoProj 0 &
wait

echo "=== MPO-NoProj model=2 (tinyllama11b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 1 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 2 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 3 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 4 MPO-NoProj 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 5 MPO-NoProj 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 6 MPO-NoProj 0 &
wait

# ================================================================
# Experiment 2: MPO-LengthDiag (variant 39)
# ================================================================
echo "=== MPO-LengthDiag model=3 (qwen3b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 1 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 2 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 3 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 4 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 5 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 6 MPO-LengthDiag 0 &
wait

echo "=== MPO-LengthDiag model=4 (llama7b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 1 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 2 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 3 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 4 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 5 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 6 MPO-LengthDiag 0 &
wait

echo "=== MPO-LengthDiag model=1 (qwen05b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 1 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 2 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 3 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 4 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 5 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 6 MPO-LengthDiag 0 &
wait

echo "=== MPO-LengthDiag model=2 (tinyllama11b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 1 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 2 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 3 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 4 MPO-LengthDiag 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 5 MPO-LengthDiag 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 6 MPO-LengthDiag 0 &
wait

# ================================================================
# Experiment 3: MPO-LengthOnly (variant 40)
# ================================================================
echo "=== MPO-LengthOnly model=3 (qwen3b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 1 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 2 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 3 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 4 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 5 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 6 MPO-LengthOnly 0 &
wait

echo "=== MPO-LengthOnly model=4 (llama7b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 1 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 2 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 3 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 4 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 5 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 6 MPO-LengthOnly 0 &
wait

echo "=== MPO-LengthOnly model=1 (qwen05b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 1 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 2 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 3 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 4 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 5 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 6 MPO-LengthOnly 0 &
wait

echo "=== MPO-LengthOnly model=2 (tinyllama11b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 1 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 2 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 3 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 4 MPO-LengthOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 5 MPO-LengthOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 6 MPO-LengthOnly 0 &
wait

# ================================================================
# Experiment 4: MPO-NormOnly (variant 41)
# ================================================================
echo "=== MPO-NormOnly model=3 (qwen3b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 1 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 2 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 3 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 4 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 5 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 6 MPO-NormOnly 0 &
wait

echo "=== MPO-NormOnly model=4 (llama7b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 1 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 2 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 3 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 4 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 4 5 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 4 6 MPO-NormOnly 0 &
wait

echo "=== MPO-NormOnly model=1 (qwen05b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 1 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 2 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 3 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 4 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 5 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 6 MPO-NormOnly 0 &
wait

echo "=== MPO-NormOnly model=2 (tinyllama11b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 1 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 2 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 3 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 4 MPO-NormOnly 0 &
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 5 MPO-NormOnly 0 &
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 6 MPO-NormOnly 0 &
wait

echo "🎉 ALL 96 ABLATION JOBS FINISHED"