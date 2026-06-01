# qwen3b ultrallama SFT Cache Check

## Verification command
```
ls .cache/thinng/*ultrallama*qwen3b*sft*/LATEST/policy.pt
```

## Result

| Item            | Value |
|-----------------|-------|
| Folder          | `.cache/thinng/ultrallama_qwen3b_sft_2026-05-27_12-51-52_950825` |
| Size            | 12G |
| Modified        | 2026-05-27 13:25:30 |
| LATEST/policy.pt    | **YES** |
| LATEST/optimizer.pt | YES |
| LATEST/scheduler.pt | YES |
| config.yaml         | YES |
| **Status**          | **COMPLETE** |

## Conclusion

The SFT checkpoint for `(qwen3b, ultrallama)` is present and complete.
The previous run failure was due to the incomplete folder having been present
at submission time. That folder has since been removed and replaced by this
valid one.

Safe to rerun SimPO on `(qwen3b, ultrallama)`.
