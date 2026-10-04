# Training Data — kpiya/kas-4b

## Overview

Training data for the `kpiya/kas-4b` Decision Index fine-tune.

- **Total rows**: 12,700
- **Format**: Hugging Face chat-template messages (system / user / assistant)
- **Builder**: `kas_engine/build_train_data.py` (seed `0xA55`)
- **Contamination rule**: any row whose `id` matches a `group_id` in the Decision Index v0.2.1 suite (both `selected-rows.jsonl.gz` and `added-rows.jsonl.gz`) is excluded.
- **Blocklist size**: 148,183 group_ids

## Sources

All training rows come from benchmark data that was normalized as part of the suite rebuild but **not selected** into the test suite (overflow rows from capped benchmarks). No external datasets were downloaded for training.

| Benchmark | Rows sampled | Pool available | Option count |
|-----------|-------------|----------------|--------------|
| RouterBench-0shot | 2,000 | 58,348 | 11 |
| BPoMP-original-limerick | 1,500 | 134,005 | 2 |
| SGD-service-given-intent | 1,500 | 43,616 | 2–5 |
| Amazon-ESCI | 1,500 | 633,016 | 4 |
| RouterBench-5shot | 500 | 58,322 | 11 |
| Habermas-consensus | 800 | 1,676 | 4–5 |
| CLadder | 800 | 5,112 | 2 |
| GSM8K-4choice | 800 | 1,319 | 4 |
| ChessBench-legal-move | 500 | 61,833 | varies |
| ACOS-category-sentiment | 600 | 318,945 | 2 |
| CFColor-preference | 500 | 6,821 | 2 |
| ForecastBench-binary | 500 | 10,139 | 2 |
| CRUXEval-output-choice | 400 | 570 | 4 |
| BRIGHT-retrieval | 300 | 1,059 | 2 |
| ToolRet-retrieval | 300 | 7,699 | 2 |
| POP909-chord-pitch-class | 200 | 75,582 | varies |

## Prompt format

Each training example is a three-message conversation:

```
system:    "You are a decision engine. Read the JSON object carefully,
            then output exactly one option key from the options field.
            Output only the key, nothing else."

user:      {"state": {...}, "question": "...", "options": {"A": "...", "B": "...", ...}}

           Let me repeat that:

           {"state": {...}, "question": "...", "options": {"A": "...", "B": "...", ...}}

assistant: <single uppercase label, e.g. "C">
```

- Original option keys (`option_0`, `yes`, `ReserveRestaurant`, etc.) are mapped to single-token uppercase labels (A–Z then AA, AB, …) so the answer is always one token.
- The user message includes the JSON payload twice separated by `\n\nLet me repeat that:\n\n` (the prompt-repeat trick; Scion measured +1.3 pts at 4B).
- `state` is omitted from the payload when empty.

## Contamination policy

1. Build blocklist from all `_evaluation.group_id` values in `selected-rows.jsonl` and `added-rows.jsonl.gz` of the Decision Index v0.2.1 suite rebuild.
2. Skip any normalized row whose `id` matches a blocklist entry before sampling.
3. Blocklist is saved to `work/train/blocklist.json` for auditing.

## What is NOT in the training data

- No rows from benchmarks where the suite consumed all available test-split rows (MMLU, HellaSwag, ARC, BANKING77, ANLI, WinoGrande, BFCL, CLINC150+OOS, NLI4CT, ContractNLI, SimpleBench, HLE, GPQA, MuSR, VAST, SATA-Bench, iSarcasmEval).
- No synthetic data, no external crawling, no human annotation.
- No rows where the expected answer was `null` (BRIGHT/ToolRet retrieval rows without ground truth).
