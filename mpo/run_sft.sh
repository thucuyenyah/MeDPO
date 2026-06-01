#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU ưu tiên: 7 > 5
# GPU 7: free hoàn toàn
# GPU 5: còn ~62GB

# ===============================
# model 3 - qwen3b (ưu tiên)
# ===============================
echo "=== SFT model=3 (qwen3b) ==="
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 1 DPO 1 &   # hh
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 2 DPO 1 &   # shp
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 3 DPO 1 &   # pku
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 4 DPO 1 &   # ultrabin
wait
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 5 DPO 1 &   # ultrallama
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 6 DPO 1 &   # ultragemma
wait



# # ===============================
# # model 1 - qwen05b
# # ===============================
# echo "=== SFT model=1 (qwen05b) ==="
# CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 1 DPO 1 &   # hh
# CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 2 DPO 1 &   # shp
# wait
# CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 3 DPO 1 &   # pku
# CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 4 DPO 1 &   # ultrabin
# wait
# CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 5 DPO 1 &   # ultrallama
# CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 6 DPO 1 &   # ultragemma
# wait

# # ===============================
# # model 2 - tinyllama11b
# # ===============================
# echo "=== SFT model=2 (tinyllama11b) ==="
# CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 1 DPO 1 &   # hh
# CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 2 DPO 1 &   # shp
# wait
# CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 3 DPO 1 &   # pku
# CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 4 DPO 1 &   # ultrabin
# wait
# CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 5 DPO 1 &   # ultrallama
# CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 6 DPO 1 &   # ultragemma
# wait

echo "🎉 ALL 24 SFT JOBS FINISHED"