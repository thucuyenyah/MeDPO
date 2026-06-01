# GitHub Preparation Status

Generated: 2026-05-27 (updated for GPU selection)

---

## 1. Files copied from source

| Source | Destination | Notes |
|--------|-------------|-------|
| `fd/train.py` | `mpo/train.py` | SimPO reference-model fix already present |
| `fd/trainers.py` | `mpo/trainers.py` | All loss implementations (DPO, SimPO, TDPO, TIS, CDPO, frontdoor, fddpo) |
| `fd/utils.py` | `mpo/utils.py` | |
| `fd/preference_datasets.py` | `mpo/preference_datasets.py` | |
| `fd/frontdoor.py` | `mpo/frontdoor.py` | FrontdoorMediatorCapture / FrontdoorSteering |
| `fd/fddpo.py` | `mpo/fddpo.py` | FABE-DPO hidden-state losses (MPO core) |
| `fd/requirements.txt` | `mpo/requirements.txt` | |
| `fd/slurm_env.sh` | `mpo/slurm_env.sh` | Self-referential PROJECT_ROOT |
| `fd/LICENSE` | `mpo/LICENSE` | |
| `fd/config/config.yaml` | `mpo/config/config.yaml` | |
| `fd/config/loss/dpo.yaml` | `mpo/config/loss/dpo.yaml` | |
| `fd/config/loss/sft.yaml` | `mpo/config/loss/sft.yaml` | |
| `fd/config/model/qwen05b.yaml` | `mpo/config/model/qwen05b.yaml` | |
| `fd/config/model/tinyllama11b.yaml` | `mpo/config/model/tinyllama11b.yaml` | |
| `fd/config/model/qwen3b.yaml` | `mpo/config/model/qwen3b.yaml` | |
| `fd/config/model/llama7b.yaml` | `mpo/config/model/llama7b.yaml` | From `fd/`, float32 policy dtype |
| `fd/betaDPO/train.py` | `mpo/betaDPO/train.py` | |
| `fd/betaDPO/trainers.py` | `mpo/betaDPO/trainers.py` | beta_DPO-specific DPO loss |
| `fd/betaDPO/utils.py` | `mpo/betaDPO/utils.py` | |
| `fd/betaDPO/preference_datasets.py` | `mpo/betaDPO/preference_datasets.py` | |
| `fd/betaDPO/config/*` | `mpo/betaDPO/config/*` | All model + loss configs |
| `fd/betaDPO/config/model/llama7b.yaml` | `mpo/betaDPO/config/model/llama7b.yaml` | Already present in betaDPO source |

---

## 2. Files created (new)

| File | Description |
|------|-------------|
| `mpo/run_all.sh` | Refactored entry point with method-name interface |
| `mpo/commands.txt` | Complete reproduction commands for all 4×6×14 jobs |
| `mpo/README.md` | Clean public README |
| `mpo/.gitignore` | Excludes caches, checkpoints, wandb, HF, runs, logs |
| `mpo/scripts/smoke_test.sh` | Argument-parsing smoke test (48 checks, all pass) |
| `mpo/REPRODUCIBILITY_GAPS.md` | Documents known gaps |
| `mpo/PREPARE_GITHUB_STATUS.md` | This file |

---

## 3. Files excluded (not copied)

The following were intentionally excluded from the public release:

- `fd/runs/`, `fd/logs/`, `fd/outputs/`, `fd/.cache/` — generated outputs
- `fd/wandb/` — W&B logs
- `fd/conda_envs/` — conda environments
- `fd/analysis/`, `fd/pre_subs/`, `fd/refs/`, `fd/best_papers/` — internal analysis
- `fd/scripts/submit_*.sh`, `fd/scripts/smoke_*.sh` — internal job submission scripts
- `fd/scripts/monitor_*.sh` — internal monitoring scripts
- `fd/*.md` (BASELINE_*.md, FABE_*.md, IMPLEMENTATION_PLAN.md, etc.) — internal notes
- `fd/get_result.py`, `fd/result*.csv`, `fd/budget_runtime.py`, etc. — result analysis tools
- `fd/tests/`, `fd/tmp/`, `fd/added/`, `fd/winrate/` — miscellaneous internal dirs
- `fd/betaDPO/runs/`, `fd/betaDPO/logs/`, `fd/betaDPO/outputs/` — betaDPO generated outputs
- All `*.pyc`, `__pycache__/` — Python cache

---

## 4. Method mapping (internal → public)

| Internal name | Public name | run_all.sh method | Source variant |
|--------------|-------------|-------------------|----------------|
| originaldpo | DPO | `DPO` | variant=9 |
| bDPO | betaDPO | `betaDPO` | betaDPO subdir |
| SimPO | SimPO | `SimPO` | variant=60 |
| TDPO | TDPO | `TDPO` | variant=61 |
| TISDPO | TIS | `TIS` | variant=62 |
| CDPOBackdoor | CDPO | `CDPO` | variant=65 |
| fdDPOv7CW / CausalWalk | CW | `CW` | variant=29 |
| fdFABETS | MPO-TS | `MPO-TS` | variant=36 |
| fdFABEDL | MPO-Dual | `MPO-Dual` | variant=37 |
| fdFABEMA | MPO-EMA | `MPO-EMA` | variant=35 |
| fdDPOv12LN | MPO-LN | `MPO-LN` | variant=34 |
| fdDPOv10Safe | MPO-Safe | `MPO-Safe` | variant=32 |
| fdDPOv9Conf | MPO-Conf | `MPO-Conf` | variant=31 |
| fdDPOv11CSafe | MPO-ConfSafe | `MPO-ConfSafe` | variant=33 |

---

## 5. Model mapping

| model_idx | Model name | HuggingFace ID |
|-----------|-----------|----------------|
| 1 | qwen05b | `Qwen/Qwen2.5-0.5B` |
| 2 | tinyllama11b | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| 3 | qwen3b | `Qwen/Qwen2.5-3B` |
| 4 | llama7b | `huggyllama/llama-7b` |

---

## 6. betaDPO integration status

**Status: DONE**

betaDPO is integrated into `run_all.sh` via `is_betadpo=1` flag. When `method=betaDPO`,
`run_all.sh` changes directory into `betaDPO/` and calls `betaDPO/train.py` with the
beta-DPO-specific parameters (`loss.mode_loss=beta_DPO loss.mode_weight=0.2 loss.a=0.6`).

betaDPO shares the SFT checkpoint with DPO (same checkpoint discovery logic).
betaDPO keeps its own separate `betaDPO/config/` because its `dpo.yaml` has different
fields (`mode_loss`, `mode_weight`, `a`) incompatible with the main `config/loss/dpo.yaml`.

---

## 7. LLaMA-7B integration status

**Status: DONE**

- `config/model/llama7b.yaml` added (from `fd/config/model/llama7b.yaml`)
- `betaDPO/config/model/llama7b.yaml` added (from `fd/betaDPO/config/model/llama7b.yaml`)
- `run_all.sh` model array now includes `llama7b` as index 4
- Layer configs in `run_all.sh` use `qwen3b|llama7b` grouping (both use layer 31)
- All 14 methods have been verified to have llama7b layer configs

Layer assignment for llama7b:
- Single-layer methods (CW, MPO-*): layer=31
- MPO-Dual: low_layer=20, layer=31
- Frontdoor variants (not in public interface): layer=30 (same as qwen3b)

---

## 8. commands.txt and submission helpers

`mpo/commands.txt` revised to reflect dependency-based structure and GPU selection:
- Header documents GPU requirements (H100/H200 only, no A100) and `[gpu_type]` argument
- Recommended section shows all four invocation forms (auto/h100/h200/dry-run)
- Section A: Manual step-by-step with `GPU=--gpus=h100:1` variable; all sbatch calls include `--partition=gpu-large $GPU`
- Section B: 56 method/model sbatch commands with `--partition=gpu-large $GPU --dependency=afterok:...`
- Section C: Monitoring commands
- Section D: Quick single-pair test (includes `--partition=gpu-large $GPU`)

New scripts:
| Script | Purpose |
|--------|---------|
| `scripts/submit_full_reproduction.sh` | Master script: GPU selection (auto/h100/h200), submits all 80 jobs, writes manifest |
| `scripts/run_method_model_all_datasets.sh` | SLURM job script: runs one method×model over all 6 datasets sequentially (no GPU selection — receives it from submission script via command-line override) |

GPU selection in `scripts/submit_full_reproduction.sh`:
- Accepts `[auto|h100|h200]` and `[--dry-run]` in any order
- `auto`: queries cluster with `sinfo`, prefers H200 if idle/mix nodes exist, falls back to H100
- Prints selected GPU type and reason before submitting
- Passes `--partition=gpu-large --gpus=<type>:1` to every sbatch call via `GPU_FLAGS` array
- Manifest gains `gpu_type` as 8th column

`run_all.sh` and `run_method_model_all_datasets.sh` have **no GPU selection logic**:
- Both have `#SBATCH --partition=gpu-large` but no `#SBATCH --gpus=...` line
- GPU type is supplied by the submission script on the sbatch command line (overrides SBATCH directives)

Total SLURM jobs submitted:
- 24 SFT jobs (4 models × 6 datasets), job names S1D1 … S4D6
- 56 method jobs (14 methods × 4 models), job names DPO_M1 … MCSF_M4

Dependency rule: method job `<PREFIX>_M<m>` depends on `afterok:S<m>D1:...:S<m>D6`.
All SFT and method jobs are submitted simultaneously; SLURM queues method jobs to wait.

Manifest written to: `runs/reproduction_jobs.tsv`
Columns: stage, method, model_idx, dataset_idx, job_name, job_id, dependency, gpu_type

---

## 9. Smoke test result

**All 69 checks passed.**

```
=== Results: 69 passed, 0 failed ===
```

Checks cover:
- No-arg usage message
- Invalid model_idx / dataset_idx rejection
- Invalid / old method names rejection (TISDPO, CausalWalk, bDPO, originaldpo, CDPOBackdoor)
- All 14 public method names present in run_all.sh
- All 12 old internal names absent from case labels
- Shell syntax (bash -n) for run_all.sh, submit_full_reproduction.sh, run_method_model_all_datasets.sh
- betaDPO subdirectory structure (6 files)
- Model configs (4 models)
- submit_full_reproduction.sh and run_method_model_all_datasets.sh present and syntactically valid
- All 16 job name prefixes ≤8 characters
- Inner loop covers all 6 datasets

---

## 10. Remaining reproducibility gaps

See `REPRODUCIBILITY_GAPS.md` for full details. Summary:

1. **Win-rate evaluation**: not included (external judge API, pending)
2. **LLaMA-7B memory**: may require FSDPTrainer on GPUs < 80 GB
3. **betaDPO hyperparameters**: tuned values (mode_weight=0.2, a=0.6) are already set as defaults
