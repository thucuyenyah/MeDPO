# SIM_M3 Rerun Plan

## Failed datasets

| Model  | Dataset    | Reason                                                        |
|--------|------------|---------------------------------------------------------------|
| qwen3b | ultrallama | Archive step failed at run_all.sh:412 — SFT checkpoint path  |
|        |            | `.cache/thinng/ultrallama_qwen3b_sft_*/LATEST/policy.pt`     |
|        |            | does not exist. Training completed (FINAL eval = 50.39%)     |
|        |            | but `Done:` line never written. Exit 127 from missing file.  |

## Rerun scope

Only 1/6 datasets failed → rerun only the failed dataset:

    (model=qwen3b, dataset=ultrallama)

Do NOT rerun the full SIM_M3 job. The other 5 datasets have valid results.

## Pre-rerun checklist

1. Verify the SFT checkpoint exists before submitting:
       ls .cache/thinng/ultrallama_qwen3b_sft_*/LATEST/policy.pt
   If missing, the same archive step will fail again.

2. If checkpoint is missing, the SFT job for (qwen3b, ultrallama) = S3D5
   needs to be rerun first (original job 93887, now completed).
   Check whether checkpoint was saved:
       ls .cache/thinng/*ultrallama*qwen3b*sft*/LATEST/

3. Once SFT checkpoint is confirmed, submit a targeted single-dataset run:
       sbatch --partition=gpu-large --gpus=h100:1 \
              --job-name=SIM_M3_ultrallama_rerun \
              run_all.sh 3 5 SimPO 0

## Notes

- The training weights for this run ARE written to:
      .cache/thinng/ultrallama_qwen3b_SimPO_2026-05-27_20-41-58_079357/LATEST/
  The failure was purely in the archiving post-step, not in training itself.
- Accuracy from the partial run (50.39%) is numerically valid but cannot be
  used without the Done line confirmation — rerun is required.
