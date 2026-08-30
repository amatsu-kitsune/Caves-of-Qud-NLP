"""
Build the VERIFIER dataset from the TIERED 100-text set (train/val/test),
REPLACING the 300-story source. Run `split_tiered.py` FIRST (adds the `split` field).

Each tiered example carries its gold triple(s) and (after the relation-only synonymy
rewrite) names its subjects/objects canonically, so the evidence for a triple is that
example's own text. For every gold triple we mark its span(s) and emit VALID + INVALID
negatives (hard / direction / domain-range), exactly like build_verifier_from_stories.py.
Split follows the tiered split -> no leakage. The span-marking + negative-generation and
ontology helpers are REUSED from build_verifier_from_stories.py so the two builders can't
drift.

Inputs  (generated/): qud_tiered_annotated.jsonl (with `split`), gold_triples.jsonl,
                      ontology_interface.json
Outputs (generated/): verifier_{train,val,test,all}.jsonl, verifier_stats.json

Usage: python build_verifier_from_tiered.py [--no-datatype] [--seed 13]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import build_verifier_from_stories as bvs

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-datatype", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    bvs.DT_ANCHOR.update({
        "level": lambda v: rf"\blevel[\s-]{re.escape(v)}\b",
        "Value": lambda v: rf"\bworth\s+{re.escape(v)}\b|\b{re.escape(v)}\s+drams\b",
        "Weight": lambda v: rf"\b(?:weighs|weight of|weighing)\s+{re.escape(v)}\b",
    })

    examples = bvs.load_jsonl(GEN / "qud_tiered_annotated.jsonl")
    if not all("split" in e for e in examples):
        raise SystemExit("qud_tiered_annotated.jsonl has no 'split' field -> "
                         "run `python split_tiered.py` first")
    gold = bvs.load_jsonl(GEN / "gold_triples.jsonl")
    iface = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))

    relmeta = {r["short"]: r for r in iface["relation_vocabulary"]}
    obj_props = [r["short"] for r in iface["relation_vocabulary"] if r["kind"] == "object_property"]
    data_props = [r["short"] for r in iface["relation_vocabulary"] if r["kind"] == "datatype_property"]
    ancestors, ind_types = iface["class_ancestors"], iface["individual_types"]

    label_of = {}
    for t in gold:
        label_of[t["subject"]] = t["subject_label"]
        if t["kind"] == "object_property":
            label_of[t["object"]] = t["object_label"]
    label_of.update(bvs.SURFACE_OVERRIDES)
    all_labels = sorted(set(label_of.values()), key=len, reverse=True)

    def closure(i):
        out = set()
        for ty in ind_types.get(i, [i]):
            out |= set(ancestors.get(ty, [ty]))
        return out

    def compat(p, s, o):
        dom, rng_ = set(relmeta[p]["domain"]), set(relmeta[p]["range"])
        return ((not dom) or dom & closure(s)) and ((not rng_) or rng_ & closure(o))

    def compat_dom(p, s):
        dom = set(relmeta[p]["domain"])
        return (not dom) or bool(dom & closure(s))

    gold_obj = {(t["subject"], t["predicate"], t["object"]) for t in gold if t["kind"] == "object_property"}
    pair_preds = defaultdict(set)
    for s, p, o in gold_obj:
        pair_preds[(s, o)].add(p)

    rows, seen = [], set()

    def add(text_marked, label, pred, s, o, neg, kind, split):
        key = (text_marked, label, pred)
        if key in seen:
            return
        seen.add(key)
        rows.append({"text": f"[REL] {pred} [/REL] {text_marked}", "label": label,
                     "candidate_relation": pred, "subject": s, "object": o,
                     "neg_type": neg, "rel_kind": kind, "split": split})

    obj_misses = dt_misses = 0
    for ex in examples:
        split = ex["split"]
        txt = ex["text"]
        for tr in ex["triples"]:
            s, p = tr["subject"], tr["predicate"]
            if tr["kind"] == "datatype_property":
                if args.no_datatype:
                    continue
                v = str(tr["object"])
                valid = bvs.mark_literal(txt, label_of[s], p, v, all_labels)
                if valid is None:
                    dt_misses += 1
                    continue
                add(valid, "VALID", p, s, "LiteralValue", None, "datatype", split)
                pool = [q for q in data_props if q != p and compat_dom(q, s)]
                if pool:
                    add(valid, "INVALID", rng.choice(pool), s, "LiteralValue", "hard", "datatype", split)
                continue
            o = tr["object"]
            ls, lo = label_of[s], label_of[o]
            valid = bvs.mark_pair(txt, ls, lo, all_labels)
            if valid is None:
                obj_misses += 1
                continue
            add(valid, "VALID", p, s, o, None, "object", split)
            hard = [q for q in obj_props if q != p and compat(q, s, o) and q not in pair_preds[(s, o)]]
            if hard:
                add(valid, "INVALID", rng.choice(hard), s, o, "hard", "object", split)
            if (o, p, s) not in gold_obj:
                rev = bvs.mark_pair(txt, lo, ls, all_labels)
                if rev is not None:
                    add(rev, "INVALID", p, o, s, "direction", "object", split)
            else:
                dr = [q for q in obj_props if not compat(q, s, o)]
                if dr:
                    add(valid, "INVALID", rng.choice(dr), s, o, "dr", "object", split)

    def base(t):
        return re.sub(r"\[/?(E1|E2)\]", "", re.sub(r"^\[REL\] [^\]]+ \[/REL\] ", "", t))
    test_bases = {base(r["text"]) for r in rows if r["split"] == "test"}
    val_bases = {base(r["text"]) for r in rows if r["split"] == "val"}

    def _keep(r):
        b = base(r["text"])
        if r["split"] == "train" and (b in val_bases or b in test_bases):
            return False
        if r["split"] == "val" and b in test_bases:
            return False
        return True

    before = len(rows)
    rows = [r for r in rows if _keep(r)]
    purged = before - len(rows)

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    for split in ("train", "val", "test"):
        with (GEN / f"verifier_{split}.jsonl").open("w", encoding="utf-8") as h:
            for r in by_split[split]:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (GEN / "verifier_all.jsonl").open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "source": f"qud_tiered_annotated.jsonl ({len(examples)} texts, 70/15/15 split)",
        "total_rows": len(rows),
        "label_distribution": dict(Counter(r["label"] for r in rows)),
        "neg_type_distribution": dict(Counter(r["neg_type"] for r in rows if r["neg_type"])),
        "rel_kind_distribution": dict(Counter(r["rel_kind"] for r in rows)),
        "split_distribution": {k: len(v) for k, v in by_split.items()},
        "split_label": {k: dict(Counter(r["label"] for r in v)) for k, v in by_split.items()},
        "distinct_candidate_relations": len(set(r["candidate_relation"] for r in rows)),
        "object_triple_marking_misses": obj_misses,
        "datatype_triple_marking_misses": dt_misses,
        "train_rows_purged_for_val_test_overlap": purged,
    }
    (GEN / "verifier_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
