# kas-4b: Decision Engine Run Log

**Project:** kpiya/kas-4b — Decision Index 0.2.1 submission  
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
- Model name: kpiya/kas-4b
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

| Date | Model | Config | Sample | Index | Per-area | Latency (ms) | Cost |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | kpiya/kas-4b | LoRA r64 α128 2ep lr1e-4 bs16 4096tok | 150,759 rows | **40.1** (raw 54.18) | K&R 39.5 / Lang 35.2 / Ret 40.4 / Tools 49.0 / Arts 37.7 | median 39ms p95 546ms | ~$8–10 est |

## HF Jobs

| Job ID | Status | Notes |
|---|---|---|
| 6ac1f17a404719ba37646e06 | ERROR | Failed: transformers too old for qwen3_5 arch |
| 6ac1f21b404719ba37646f11 | ERROR | ImportError: DataCollatorForCompletionOnlyLM |
| 6ac1f2abfbc85ba6823972a1 | CANCELLED | No causal-conv1d, ~24s/step, 8.9h est |
| 6ac1fecd404719ba376483d1 | ERROR | causal-conv1d build failed, runtime image no nvcc |
| 6ac1ff70404719ba376484d6 | ERROR | CUDA 12.8 vs PyTorch 2.8 compiled for CUDA 13.0 |
| 6ac20029fbc85ba682398a76 | COMPLETED | v6: Qwen3-4B (pure transformer, no SSM kernels) — 1h 24m |
| 6ac21573404719ba3764a773 | COMPLETED | Phase 5 calibration: 51.07% acc (choice 48.59%, noul 76.19%), T=1.0 |
| 6ac22286404719ba3764bb3f | CANCELLED | Phase 6 attempt 1: kas_engine not installed (source_root was kas0 not decision-index) |
| 6ac222fa404719ba3764bbd5 | CANCELLED | Phase 6 attempt 2: hf download needs --token $HF_TOKEN for private repo |
| 6ac22356404719ba3764bcc0 | COMPLETED | Phase 6 full eval: Decision Index **40.1** (raw 54.18) — 150,759 rows, 23,570s (~6.55h) |

