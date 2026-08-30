"""
Re-split the existing story corpus into TRAIN / VAL / TEST = 70 / 15 / 15.

Operates on the already-generated text (does NOT regenerate stories), so it is
deterministic and non-destructive to wording. The split is STRATIFIED by the
`complex` flag so each split keeps a comparable share of hard stories.

Rewrites:
  generated/story_corpus.jsonl            (only the "split" field changes)
  data-input/{train,val,test}/story_*.txt
  generated/story_corpus_stats.json

After this you MUST rebuild the verifier dataset and RETRAIN (old models were
trained on a different split -> would leak into the new test set):
  python build_verifier_from_stories.py
  python bench_train_all.py --force

Usage: python resplit_corpus.py [--seed 7] [--train 0.70] [--val 0.15]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
DATA = BASE / "data-input"
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    corpus = [json.loads(l) for l in (GEN / "story_corpus.jsonl").open(encoding="utf-8") if l.strip()]

    split_of = {}
    for flag in (False, True):
        ids = sorted(s["story_id"] for s in corpus if bool(s.get("complex")) == flag)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = round(n * args.train)
        n_va = round(n * args.val)
        for i, sid in enumerate(ids):
            split_of[sid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    for s in corpus:
        s["split"] = split_of[s["story_id"]]

    with (GEN / "story_corpus.jsonl").open("w", encoding="utf-8") as h:
        for row in corpus:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    for sub in SPLITS:
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for f in (DATA / sub).glob("story_*.txt"):
            f.unlink()
    for s in corpus:
        (DATA / s["split"] / f'{s["story_id"]}.txt').write_text(s["text"] + "\n", encoding="utf-8")

    stats = {
        "total": len(corpus),
        "split": dict(Counter(s["split"] for s in corpus)),
        "split_complex": {sp: sum(s["split"] == sp and s["complex"] for s in corpus) for sp in SPLITS},
        "lines_per_split": {sp: sum(len(s["lines"]) for s in corpus if s["split"] == sp) for sp in SPLITS},
        "ratios": {"train": args.train, "val": args.val, "test": round(1 - args.train - args.val, 4)},
        "seed": args.seed,
    }
    (GEN / "story_corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
