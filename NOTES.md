# kas-4b: Decision Engine Run Log

**Project:** kas0/kas-4b — Decision Index 0.2.1 submission  
**Base model:** Qwen/Qwen3.5-4B (Apache 2.0)  
**Upgrade path:** Qwen/Qwen3.5-9B  
**Method:** LoRA fine-tune (PEFT), merge adapter, serve via transformers  
**Prompt trick:** Send JSON prompt twice (Scion-style repeat)  
**Calibration:** Temperature scaling on dev rows only  
**Spending cap:** $25/job

---

## Phase 0 — 2026-10-03

- Repo initialized at /Users/kashavpiya/kas0 (Python 3.13.7)
- Base model Qwen/Qwen3.5-4B confirmed: Apache 2.0, 262k context, Hub id `Qwen/Qwen3.5-4B`
- Upgrade model Qwen/Qwen3.5-9B confirmed: Apache 2.0, 262k context, Hub id `Qwen/Qwen3.5-9B`
- HLE terms accepted by user; HF billing configured
- Model name: kas0/kas-4b
- Spending cap: $25 per job

### Reference benchmarks (from plan, 2026-10-03)

| Model | Index | Source |
|---|---|---|
| Jev (oracle) | 57.9 | board |
| Scion v4 (9B, unofficial) | 45.9 | board |
| Tev1-4B | 29.2 | board |

### Training recipe references

**Tev1-4B:** rank 8, 1 epoch, LR 5e-5, 2048 token limit, 37,840 rows, ~$17 total  
**Scion v4 (9B):** rank 64, 2 epochs, LR 1e-4, ~51k rows, temperature T=1.4–1.9  
**Scion prompt repeat gain:** +1.3 pts at 4B, +4.6 pts at 9B

---

## Runs

_(will be filled in from Phase 2 onward)_

| Date | Model | Config | Sample | Index | Per-area | Latency (ms) | Cost |
|---|---|---|---|---|---|---|---|

