"""Phase 3: Build training data for kas0/kas-4b fine-tuning.

Sources: unselected rows from already-normalized benchmark files.
Contamination rule: any row whose id appears in the suite's group_id
set is excluded — guaranteed by comparing against both selected-rows
and added-rows from the rebuilt suite artifacts.

Output:
  work/train/train.jsonl      — (messages, meta) pairs for SFT
  work/train/blocklist.json   — suite group_ids used as contamination filter
  work/train/stats.json       — per-benchmark sampling counts
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-index"))

from decision_index.suite.io import read_jsonl

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM_DIR = os.path.join(ROOT, "work", "artifacts", "benchmark-suite", "normalized")
SUITE_V1 = os.path.join(ROOT, "work", "artifacts", "benchmark-suite",
                         "release-v1-rebuilt", "selected-rows.jsonl")
SUITE_V2_GZ = os.path.join(ROOT, "work", "artifacts", "benchmark-suite",
                             "release-v2-rebuilt", "added-rows.jsonl.gz")
OUT_DIR = os.path.join(ROOT, "work", "train")
SEED = 0xA55

SYSTEM = (
    "You are a decision engine. Read the JSON object carefully, "
    "then output exactly one option key from the options field. "
    "Output only the key, nothing else."
)

# ── Sampling targets ────────────────────────────────────────────────────────
# (filename_stem, max_rows)
TARGETS: list[tuple[str, int]] = [
    ("RouterBench-0shot",          2000),
    ("BPoMP-original-limerick",    1500),
    ("SGD-service-given-intent",   1500),
    ("Amazon-ESCI",                1500),
    ("RouterBench-5shot",           500),
    ("Habermas-consensus",          800),
    ("CLadder",                     800),
    ("GSM8K-4choice",               800),
    ("ChessBench-legal-move",       500),
    ("ACOS-category-sentiment",     600),
    ("CFColor-preference",          500),
    ("ForecastBench-binary",        500),
    ("CRUXEval-output-choice",      400),
    ("BRIGHT-retrieval",            300),
    ("ToolRet-retrieval",           300),
    ("POP909-chord-pitch-class",    200),
]

# ── Label helpers ───────────────────────────────────────────────────────────

def _label_list() -> list[str]:
    labels = list(string.ascii_uppercase)
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            labels.append(a + b)
            if len(labels) >= 255:
                return labels
    return labels


LABELS = _label_list()


def _build_payload(state, q_type, q_instructions, q_criteria, n_opts: int):
    """Return (payload_dict, ordered_original_keys)."""
    labels = LABELS[:n_opts]
    if q_type == "choice":
        keys = list(q_criteria)
        options = {}
        for lbl, key in zip(labels, keys):
            desc = q_criteria[key]
            options[lbl] = desc if desc is not None else key
    else:
        keys = ["false", "true"]
        options = {labels[0]: "No", labels[1]: "Yes"}
    payload: dict = {}
    if state not in ("", None, {}, []):
        payload["state"] = state
    payload["question"] = q_instructions
    payload["options"] = options
    return payload, keys


def row_to_messages(row: dict) -> list[dict] | None:
    """Convert a normalized row to a list of training (messages, answer_label) tuples.

    Returns None if the row has no scoreable question (all expected=None).
    Each tuple is (messages_list, answer_label_str, question_key).
    """
    state = row.get("state", {})
    questions = row.get("questions", {})
    expected = row.get("expected", {})
    results = []

    for qk, q in questions.items():
        exp_val = expected.get(qk)
        if exp_val is None:
            continue  # no ground truth — skip

        q_type = q.get("type", "choice")
        q_instr = q.get("instructions", "")
        q_crit = q.get("criteria", {})

        if q_type not in ("choice", "noul"):
            continue

        n_opts = len(q_crit) if q_type == "choice" else 2
        if n_opts < 2:
            continue

        payload, orig_keys = _build_payload(state, q_type, q_instr, q_crit, n_opts)

        # Map expected original key → label
        try:
            idx = orig_keys.index(exp_val)
        except ValueError:
            continue  # expected not found in options
        label = LABELS[idx]

        body = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
        user_content = body + "\n\nLet me repeat that:\n\n" + body

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label},
        ]
        results.append((messages, label, qk))

    return results if results else None


# ── Main ────────────────────────────────────────────────────────────────────

def build_blocklist() -> set[str]:
    ids: set[str] = set()
    for r in read_jsonl(SUITE_V1):
        ids.add(r["_evaluation"]["group_id"])
    with gzip.open(SUITE_V2_GZ, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["_evaluation"]["group_id"])
    return ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    print("Building contamination blocklist …", flush=True)
    blocklist = build_blocklist()
    print(f"  {len(blocklist):,} suite group_ids in blocklist", flush=True)

    # Save blocklist
    bl_path = os.path.join(OUT_DIR, "blocklist.json")
    with open(bl_path, "w") as f:
        json.dump(sorted(blocklist), f)
    print(f"  Saved {bl_path}", flush=True)

    stats: dict[str, dict] = {}
    train_rows: list[dict] = []

    for stem, target in TARGETS:
        fname = f"{stem}.jsonl"
        path = os.path.join(NORM_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP {fname} (not found)", flush=True)
            continue

        pool: list[dict] = []
        skipped_blocked = 0
        skipped_no_answer = 0
        for r in read_jsonl(path):
            if r["id"] in blocklist:
                skipped_blocked += 1
                continue
            tuples = row_to_messages(r)
            if not tuples:
                skipped_no_answer += 1
                continue
            # Flatten multi-question rows — each question becomes one training ex
            for messages, label, qk in tuples:
                pool.append({
                    "messages": messages,
                    "meta": {
                        "source_id": r["id"],
                        "question_key": qk,
                        "benchmark": stem,
                        "answer_label": label,
                        "n_options": len(messages[1]["content"].split('"options"')[1].split('"')[1::2]) // 2,
                    },
                })

        rng.shuffle(pool)
        sampled = pool[:target]
        train_rows.extend(sampled)

        stats[stem] = {
            "pool_size": len(pool),
            "target": target,
            "sampled": len(sampled),
            "skipped_blocked": skipped_blocked,
            "skipped_no_answer": skipped_no_answer,
        }
        print(f"  {stem}: pool={len(pool):,}  sampled={len(sampled):,}", flush=True)

    # Shuffle final list
    rng.shuffle(train_rows)

    # Write train.jsonl
    out_path = os.path.join(OUT_DIR, "train.jsonl")
    with open(out_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write stats
    stats_path = os.path.join(OUT_DIR, "stats.json")
    total = sum(s["sampled"] for s in stats.values())
    with open(stats_path, "w") as f:
        json.dump({"total": total, "seed": SEED, "benchmarks": stats}, f, indent=2)

    print(f"\nDone. {total:,} training rows → {out_path}", flush=True)
    print(f"Stats → {stats_path}", flush=True)

    # Print option count distribution
    from collections import Counter
    opt_counts: Counter = Counter()
    for row in train_rows:
        # Count options from messages — assistant content is the label
        user_msg = row["messages"][1]["content"]
        # Count option keys in first half (before repeat)
        first_half = user_msg.split("\n\nLet me repeat that:")[0]
        try:
            payload = json.loads(first_half)
            n = len(payload.get("options", {}))
            opt_counts[n] += 1
        except Exception:
            pass

    print("\nOption count distribution:")
    for k in sorted(opt_counts):
        pct = 100 * opt_counts[k] / total
        print(f"  {k} options: {opt_counts[k]:5,}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
