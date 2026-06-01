# qwen3b SFT Checkpoint Inventory

Checked: 2026-05-28

| Dataset    | checkpoint_found | checkpoint_path                                                              | status   |
|------------|-----------------|------------------------------------------------------------------------------|----------|
| hh         | YES             | `.cache/thinng/hh_qwen3b_sft_2026-05-27_12-43-52_394289/LATEST/policy.pt`       | COMPLETE |
| shp        | YES             | `.cache/thinng/shp_qwen3b_sft_2026-05-27_12-43-52_798420/LATEST/policy.pt`      | COMPLETE |
| pku        | YES             | `.cache/thinng/pku_qwen3b_sft_2026-05-27_12-44-22_468389/LATEST/policy.pt`      | COMPLETE |
| ultrabin   | YES             | `.cache/thinng/ultrabin_qwen3b_sft_2026-05-27_12-44-22_097736/LATEST/policy.pt` | COMPLETE |
| ultrallama | YES             | `.cache/thinng/ultrallama_qwen3b_sft_2026-05-27_12-51-52_950825/LATEST/policy.pt`| COMPLETE |
| ultragemma | YES             | `.cache/thinng/ultragemma_qwen3b_sft_2026-05-27_12-51-52_950502/LATEST/policy.pt`| COMPLETE |

All 6 datasets have complete SFT checkpoints (policy.pt + optimizer.pt + scheduler.pt + config.yaml, ~12G each).

**All clear — SimPO rerun for (qwen3b, ultrallama) can proceed.**
