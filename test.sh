cat run_all.sh
#!/bin/bash
set -e

# ===============================
# Environment
# ===============================
source /raid/nhdang01/miniconda3/etc/profile.d/conda.sh
conda activate dpo

export PYTHONNOUSERSITE=1
export WANDB_API_KEY=8f17474bb5e6fbb39a20e2e78dac373f97f339e6

# ===============================
# Input arguments
# ===============================
model_name_idx=$1
dataset_idx=$2
variant=$3
run_sft=$4
batch_size=8

if [ -z "$variant" ] || [ -z "$dataset_idx" ]; then
    echo "❌ Usage: bash run_all_axis.sh <model_idx> <dataset_idx> <variant> <run_sft>"
    exit 1
fi

# ===============================
# Model
# ===============================
model_names=("qwen05b" "tinyllama11b" "qwen3b")
model_name="${model_names[$((model_name_idx - 1))]}"

# ===============================
# Dataset
# ===============================
datasets=("hh" "shp" "pku")
dataset="${datasets[$((dataset_idx - 1))]}"

# ===============================
# Variant
# ===============================
case "$variant" in
    1) variant_name="dpo2samples"; num_samples="backdoor.n_mc_samples=2"; backdoor_mode="backdoor.enabled=true";;
    2) variant_name="dpo4samples"; num_samples="backdoor.n_mc_samples=4"; backdoor_mode="backdoor.enabled=true";;
    3) variant_name="dpo6samples"; num_samples="backdoor.n_mc_samples=6"; backdoor_mode="backdoor.enabled=true";;
    4) variant_name="originaldpo"; num_samples=""; backdoor_mode="backdoor.enabled=false";;
    5) variant_name="dpo1sample"; num_samples="backdoor.n_mc_samples=1"; backdoor_mode="backdoor.enabled=true";;
    *) echo "❌ Invalid variant"; exit 1;;
esac

# ===============================
# Logging (no Slurm)
# ===============================
log_dir="logs"
mkdir -p "$log_dir"
RUN_ID=$(date +%Y%m%d_%H%M%S)

log_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${run_sft}_${RUN_ID}.out"
err_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${run_sft}_${RUN_ID}.err"

exec > >(tee "$log_file") 2> >(tee "$err_file" >&2)

echo "🚀 Running $variant_name | model=$model_name | dataset=$dataset"

# ===============================
# SFT
# ===============================
if [ "$run_sft" = "1" ]; then
    python -u train.py \
        model=$model_name \
        datasets=[$dataset] \
        loss=sft \
        $backdoor_mode \
        $num_samples \
        exp_name=${dataset}_${model_name}_sft \
        gradient_accumulation_steps=2 \
        batch_size=$batch_size \
        eval_batch_size=$batch_size \
        trainer=BasicTrainer \
        sample_during_eval=false
fi


# ===============================
# DPO
# ===============================
if [ "$run_sft" = "0" ]; then
    # ===============================
    # Load checkpoint
    # ===============================
    BASE_DIR=".cache/nhdang01"
    PREFIX="${dataset}_${model_name}_sft"

    latest_suffix=$(find "$BASE_DIR" -maxdepth 1 -type d -name "${PREFIX}*" | sort | tail -n 1)

    ckpt_path="$latest_suffix/LATEST/policy.pt"

    if [ ! -f "$ckpt_path" ]; then
        echo "❌ Checkpoint not found"
        exit 1
    fi
    python -u train.py \
        model=$model_name \
        datasets=[$dataset] \
        loss=dpo \
        loss.beta=0.1 \
        $backdoor_mode \
        $num_samples \
        exp_name=${dataset}_${model_name}_${variant_name} \
        gradient_accumulation_steps=2 \
        batch_size=$batch_size \
        eval_batch_size=$batch_size \
        trainer=BasicTrainer \
        sample_during_eval=false \
        model.archive=$ckpt_path
fi

echo "✅ DONE"
cat run_all_sft.sh
#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ===============================
# Qwen 0.5B
# ===============================
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 1 4 1 &   # hh
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 1 2 4 1 &   # shp
CUDA_VISIBLE_DEVICES=1 bash run_all.sh 1 3 4 1 &   # pku

wait

# ===============================
# TinyLLaMA 1.1B
# ===============================
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 1 4 1 &   # hh
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 2 2 4 1 &   # shp
CUDA_VISIBLE_DEVICES=1 bash run_all.sh 2 3 4 1 &   # pku

wait

# ===============================
# Qwen 3B
# ===============================
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 1 4 1 &   # hh
CUDA_VISIBLE_DEVICES=5 bash run_all.sh 3 2 4 1 &   # shp
CUDA_VISIBLE_DEVICES=1 bash run_all.sh 3 3 4 1 &   # pku

wait

echo "🎉 ALL JOBS FINISHED"



cat run_all_dpo.sh
#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "==============================="
echo "🚀 START ALL DPO JOBS (FREE GPU)"
echo "   GPUs: 1, 6"
echo "==============================="

# ==================================================
# HH
# ==================================================

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 1 1 1 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 1 4 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 2 1 1 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 1 4 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 3 1 1 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 1 4 0 &
wait


# ==================================================
# SHP
# ==================================================

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 1 2 2 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 2 3 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 1 2 4 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 2 2 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 2 2 3 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 2 4 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 3 2 2 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 2 3 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 3 2 4 0 &
wait


# ==================================================
# PKU
# ==================================================

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 1 3 2 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 3 3 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 1 3 4 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 3 2 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 2 3 3 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 3 4 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 3 3 2 0 &
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 3 3 0 &
wait

CUDA_VISIBLE_DEVICES=1 bash run_all.sh 3 3 4 0 &
wait


echo "🎉 ALL DPO JOBS FINISHED (SAFE)"




