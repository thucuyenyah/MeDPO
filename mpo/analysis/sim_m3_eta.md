# SIM_M3 ETA

Job 93905 has fully terminated. No datasets are currently running.

## Final job timeline

| Dataset    | Status  | Elapsed            | Finished at (approx) |
|------------|---------|--------------------|----------------------|
| hh         | SUCCESS | 8086s (2h 14m 46s) | ~16:26 2026-05-27    |
| shp        | SUCCESS | 9508s (2h 38m 28s) | ~18:36 2026-05-27    |
| pku        | SUCCESS | 2550s (0h 42m 30s) | ~19:19 2026-05-27    |
| ultrabin   | SUCCESS | 3480s (0h 58m 0s)  | ~20:41 2026-05-27    |
| ultrallama | FAILED  | ~3383s (0h 56m 23s)| 21:38 2026-05-27 (exit 127) |
| ultragemma | SUCCESS | 3348s (0h 55m 48s) | 22:34 2026-05-27     |

Total job elapsed: 8h 27m 5s

## Rerun ETA (if ultrallama resubmitted)

Based on similar dataset sizes (ultrabin: 58m, ultragemma: 56m, ultrallama partial: 56m+):
- Estimated runtime for ultrallama rerun: ~60–70 minutes on H100
- Depends on SFT checkpoint availability (see rerun plan)

## Pending M1/M2 jobs

The 28 H100 jobs (94297–94324) for M1/M2 are all pending (Priority).
No ETA available until they begin scheduling.
