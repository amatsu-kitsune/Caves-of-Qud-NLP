"""
Assign a TRAIN / VAL / TEST = 70 / 15 / 15 split to the tiered 100-text set.

The split is written back as a `split` field on every example in
`generated/qud_tiered_annotated.jsonl`, so the
verifier-dataset builder (`build_verifier_from_tiered.py`) and the benchmark
(`bench_lib.load_units("tiered_{train,val,test}")`) can read it.

Stratified by the complexity `tier` (explicit / implicit / long_distance / nested)
via round-robin interleave, so each split keeps a comparable mix of tiers while the
GLOBAL counts land exactly on 70 / 15 / 15. Deterministic (seeded) -> re-running is
idempotent; changing the seed reshuffles and would require rebuilding + retraining.

Usage: python split_tiered.py [--seed 7] [--train 0.70] [--val 0.15]
       (test = 1 - train - val)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
ANNOT = GEN / "qud_tiered_annotated.jsonl"
NL = "\r\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    raw = ANNOT.read_text(encoding="utf-8")
    trailing = raw.endswith("\n")
    examples = [json.loads(l) for l in raw.splitlines() if l.strip()]

    by_tier = defaultdict(list)
    for e in examples:
        by_tier[e["tier"]].append(e["id"])
    rng = random.Random(args.seed)
    tiers = sorted(by_tier)
    for t in tiers:
        rng.shuffle(by_tier[t])
    order = []
    for i in range(max(len(v) for v in by_tier.values())):
        for t in tiers:
            if i < len(by_tier[t]):
                order.append(by_tier[t][i])

    n = len(order)
    n_tr = round(n * args.train)
    n_va = round(n * args.val)
    split_of = {qid: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, qid in enumerate(order)}

    for e in examples:
        e["split"] = split_of[e["id"]]

    body = NL.join(json.dumps(e, ensure_ascii=False) for e in examples)
    ANNOT.write_text(body + (NL if trailing else ""), encoding="utf-8", newline="")

    stats = {
        "source": "qud_tiered_annotated.jsonl",
        "n_examples": n,
        "ratios": {"train": args.train, "val": args.val,
                   "test": round(1 - args.train - args.val, 4)},
        "split_distribution": dict(Counter(split_of.values())),
        "split_by_tier": {sp: dict(Counter(e["tier"] for e in examples if e["split"] == sp))
                          for sp in ("train", "val", "test")},
    }
    (GEN / "qud_tiered_split_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
