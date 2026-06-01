#!/bin/bash
# submit_full_reproduction.sh
#
# Submits all SFT and preference-optimisation jobs for full reproduction.
# Run from the project root; handles all submission with correct dependencies.
#
# Usage:
#   bash scripts/submit_full_reproduction.sh [gpu_type] [--dry-run]
#
# gpu_type  (default: auto):
#   auto   – query cluster at submission time; prefer H200, then H100, then A100
#   h100   – always request H100 (gpu-large partition)
#   h200   – always request H200 (gpu-large partition)
#   a100   – A100 (gpu partition); small models only (qwen05b, tinyllama11b)
#            qwen3b and llama7b are excluded due to OOM risk.
#
# --dry-run – print all sbatch commands without submitting
#
# Flags may appear in either order.
#
# What it does (H100/H200 – all 4 models):
#   A. Submits 24 SFT jobs (4 models × 6 datasets) immediately.
#   B. Submits 56 method/model jobs (14 methods × 4 models) immediately,
#      each with --dependency=afterok:<sft_d1>:...<sft_d6> for that model.
#      Each method job loops over all 6 datasets sequentially.
#
# What it does (auto → A100 fallback):
#   A. Submits 12 SFT jobs (2 models × 6 datasets) immediately.
#   B. Submits 28 method/model jobs (14 methods × 2 models) immediately.
#   This path is used only when no H200/H100 nodes are idle/mixed.
#
# What it does (a100 – small models only):
#   A. Submits 12 SFT jobs (2 models × 6 datasets) immediately.
#   B. Submits 28 method/model jobs (14 methods × 2 models) immediately.
#   To cover qwen3b and llama7b, rerun with h100 or h200.
#
# Output:
#   runs/reproduction_jobs.tsv  – manifest with columns:
#     stage, method, model_idx, dataset_idx, job_name, job_id, dependency, gpu_type

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing  (order-independent)
# ---------------------------------------------------------------------------
GPU_TYPE="auto"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        auto|h100|h200|a100) GPU_TYPE="$arg" ;;
        --dry-run)            DRY_RUN=1       ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [auto|h100|h200|a100] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------

# Query idle/mixed H200 nodes in the gpu-large partition.
# Returns the count; 0 means none available right now.
count_available_h200() {
    sinfo -h -N -p gpu-large --format="%G %T" 2>/dev/null \
        | grep -i "h200" \
        | grep -Ei "idle|mix" \
        | wc -l \
        | tr -d '[:space:]'
}

# Query idle/mixed H100 nodes in the gpu-large partition.
# Returns the count; 0 means none available right now.
count_available_h100() {
    sinfo -h -N -p gpu-large --format="%G %T" 2>/dev/null \
        | grep -i "h100" \
        | grep -Ei "idle|mix" \
        | wc -l \
        | tr -d '[:space:]'
}

resolve_gpu() {
    local requested="$1"
    local resolved reason

    case "$requested" in
        h100)
            resolved="h100"
            reason="explicitly requested"
            ;;
        h200)
            resolved="h200"
            reason="explicitly requested"
            ;;
        a100)
            resolved="a100"
            reason="explicitly requested; small models only (qwen05b, tinyllama11b)"
            ;;
        auto)
            local h200_count
            local h100_count
            h200_count=$(count_available_h200)
            if [ "${h200_count:-0}" -gt 0 ]; then
                resolved="h200"
                reason="auto: ${h200_count} H200 node(s) idle/mix in gpu-large"
            else
                h100_count=$(count_available_h100)
                if [ "${h100_count:-0}" -gt 0 ]; then
                    resolved="h100"
                    reason="auto: no H200 nodes idle/mix in gpu-large; using ${h100_count} H100 node(s) idle/mix"
                else
                    resolved="a100"
                    reason="auto: no H200/H100 nodes idle/mix in gpu-large; falling back to A100"
                fi
            fi
            ;;
    esac

    printf "%s\t%s" "$resolved" "$reason"
}

gpu_resolve_output=$(resolve_gpu "$GPU_TYPE")
RESOLVED_GPU="${gpu_resolve_output%%$'\t'*}"
GPU_REASON="${gpu_resolve_output##*$'\t'}"

if [ "$RESOLVED_GPU" = "a100" ]; then
    GPU_FLAGS=("--partition=gpu" "--gpus=a100:1")
else
    GPU_FLAGS=("--partition=gpu-large" "--gpus=${RESOLVED_GPU}:1")
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=================================================================="
echo " MPO – Full Reproduction Submission"
echo "=================================================================="
echo " GPU type  : ${RESOLVED_GPU}  (${GPU_REASON})"
echo " SLURM     : ${GPU_FLAGS[*]}"
if [ "$RESOLVED_GPU" = "a100" ]; then
    echo " Scope     : small models only (qwen05b, tinyllama11b)"
fi
if [ "$DRY_RUN" = "1" ]; then
    echo " Mode      : DRY RUN – no jobs will be submitted"
else
    echo " Mode      : LIVE – jobs will be submitted"
fi
echo "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p runs logs

MANIFEST="${PROJECT_ROOT}/runs/reproduction_jobs.tsv"
printf "stage\tmethod\tmodel_idx\tdataset_idx\tjob_name\tjob_id\tdependency\tgpu_type\n" \
    > "$MANIFEST"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
do_sbatch() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] sbatch $*" >&2
        printf "%d" "$((10000 + RANDOM))"
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

log_row() {
    # args: stage method model_idx dataset_idx job_name job_id dependency
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "${RESOLVED_GPU}" >> "$MANIFEST"
}

method_prefix() {
    case "$1" in
        DPO)          echo "DPO"  ;;
        betaDPO)      echo "BDPO" ;;
        SimPO)        echo "SIM"  ;;
        TDPO)         echo "TDPO" ;;
        TIS)          echo "TIS"  ;;
        CDPO)         echo "CDPO" ;;
        CW)           echo "CW"   ;;
        MPO-TS)       echo "MTS"  ;;
        MPO-Dual)     echo "MDL"  ;;
        MPO-EMA)      echo "MEMA" ;;
        MPO-LN)       echo "MLN"  ;;
        MPO-Safe)     echo "MSFE" ;;
        MPO-Conf)     echo "MCF"  ;;
        MPO-ConfSafe) echo "MCSF" ;;
        *)             echo "UNK"  ;;
    esac
}

METHODS=(DPO betaDPO SimPO TDPO TIS CDPO CW \
         MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe)
if [ "$RESOLVED_GPU" = "a100" ]; then
    MODELS=(1 2)   # qwen05b, tinyllama11b only — qwen3b/llama7b risk OOM on A100
else
    MODELS=(1 2 3 4)
fi
DATASETS=(1 2 3 4 5 6)

# ---------------------------------------------------------------------------
# A. SFT jobs  (n_models × 6 datasets)
# ---------------------------------------------------------------------------
n_models=${#MODELS[@]}
n_sft=$((n_models * 6))
n_method=$((n_models * 14))
echo "=== A. Submitting SFT jobs (${n_models} × 6 = ${n_sft}) ==="

declare -A sft_jids   # sft_jids[model:dataset] = job_id

for m in "${MODELS[@]}"; do
    # qwen05b (1) and tinyllama11b (2) run on A100; heavier models use the resolved GPU
    if [[ "$m" == 1 || "$m" == 2 ]]; then
        model_gpu_flags=(--partition=gpu --gpus=a100:1)
    else
        model_gpu_flags=("${GPU_FLAGS[@]}")
    fi
    for d in "${DATASETS[@]}"; do
        jname="S${m}D${d}"
        jid=$(do_sbatch \
            "${model_gpu_flags[@]}" \
            --job-name="${jname}" \
            "${PROJECT_ROOT}/run_all.sh" "${m}" "${d}" DPO 1)
        sft_jids["${m}:${d}"]="${jid}"
        printf "  %-5s -> job %s\n" "${jname}" "${jid}"
        log_row "SFT" "DPO" "${m}" "${d}" "${jname}" "${jid}" ""
    done
done

echo ""

# ---------------------------------------------------------------------------
# B. Method/model jobs  (14 methods × n_models)
#    Depend on all 6 SFT jobs for the same model.
# ---------------------------------------------------------------------------
echo "=== B. Submitting method/model jobs (14 × ${n_models} = ${n_method}) ==="
echo "    (each loops over datasets 1–6 sequentially)"
echo ""

for method in "${METHODS[@]}"; do
    prefix=$(method_prefix "$method")
    for m in "${MODELS[@]}"; do

        # afterok dependency on all 6 SFT jobs for this model
        dep="afterok"
        for d in "${DATASETS[@]}"; do
            dep="${dep}:${sft_jids["${m}:${d}"]}"
        done

        if [[ "$m" == 1 || "$m" == 2 ]]; then
            model_gpu_flags=(--partition=gpu --gpus=a100:1)
        else
            model_gpu_flags=("${GPU_FLAGS[@]}")
        fi

        jname="${prefix}_M${m}"
        jid=$(do_sbatch \
            "${model_gpu_flags[@]}" \
            --job-name="${jname}" \
            --dependency="${dep}" \
            "${PROJECT_ROOT}/scripts/run_method_model_all_datasets.sh" \
            "${method}" "${m}")

        printf "  %-8s -> job %-8s  dep=%s\n" "${jname}" "${jid}" "${dep}"
        log_row "METHOD" "${method}" "${m}" "1-6" "${jname}" "${jid}" "${dep}"
    done
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total=$((n_sft + n_method))

echo "=================================================================="
if [ "$DRY_RUN" = "1" ]; then
    echo " DRY RUN complete – no jobs submitted."
else
    printf " Submitted %d SFT + %d method jobs = %d total.\n" \
        "${n_sft}" "${n_method}" "${total}"
fi
echo " GPU       : ${RESOLVED_GPU}  (${GPU_REASON})"
echo " Manifest  : ${MANIFEST}"
if [ "$RESOLVED_GPU" = "a100" ]; then
    echo ""
    echo " NOTE: only qwen05b and tinyllama11b were submitted (model_idx 1-2)."
    echo "       For qwen3b and llama7b, rerun with h100 or h200:"
    echo "         bash scripts/submit_full_reproduction.sh h100"
fi
echo ""
echo " Monitor:"
echo "   squeue -u \$(whoami) -o '%.18i %.8j %.8T %.10M %.6D %R'"
echo "   cat ${MANIFEST}"
echo "=================================================================="
