"""
Rebuild the VERIFIER dataset from the 300-story corpus (generate_stories.py).

Each story LINE already carries its gold triple(s), so evidence = that exact line
(rich synonym variety). For each gold triple we emit VALID + INVALID negatives
(hard / direction / domain-range). Split follows the STORY split (train/val) -> no leakage.

Inputs  (generated/): story_corpus.jsonl, gold_triples.jsonl, ontology_interface.json
Outputs (generated/): verifier_{train,val,all}.jsonl, verifier_stats.json

Usage: python build_verifier_from_stories.py [--no-datatype] [--seed 13]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"

SURFACE_OVERRIDES = {
    "Fighter_Boss": "boss-tier Fighter", "Fighter_NonBoss": "non-boss Fighter",
    "Trader_high": "high-tier merchant", "Trader_mid": "mid-tier merchant",
    "Trader_low": "low-tier merchant",
}
DT_ANCHOR = {
    "HP": lambda v: rf"\b{re.escape(v)}\s+(?:HP|hitpoints)\b",
    "level": lambda v: rf"\blevel\s+{re.escape(v)}\b",
    "strata": lambda v: rf"\bstrata\s+{re.escape(v)}\b",
    "Value": lambda v: rf"\b{re.escape(v)}\s+drams\b",
    "Weight": lambda v: rf"\b(?:weighs|weight of)\s+{re.escape(v)}\b",
}


def load_jsonl(p): return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def all_spans(text, label):
    return [(m.start(), m.end()) for m in
            re.finditer(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text, re.IGNORECASE)]


def valid_spans(text, label, labels):
    raw = all_spans(text, label)
    own = len(label)
    longer = []
    for o in labels:
        if len(o) > own:
            longer += all_spans(text, o)
    return [(s, e) for (s, e) in raw
            if not any(ls <= s and e <= le and (ls < s or e < le) for ls, le in longer)]


def mark_pair(text, la, lb, labels):
    s1, s2 = valid_spans(text, la, labels), valid_spans(text, lb, labels)
    if not s1 or not s2:
        return None
    best = None
    for a, b in s1:
        for c, d in s2:
            if b <= c or d <= a:
                k = max(b, d)
                if best is None or k < best[0]:
                    best = (k, (a, b), (c, d))
    if best is None:
        return None
    (a, b), (c, d) = best[1], best[2]
    out = text
    for s, e, tag in sorted([(a, b, "E1"), (c, d, "E2")], key=lambda x: x[0], reverse=True):
        out = out[:s] + f"[{tag}]" + out[s:e] + f"[/{tag}]" + out[e:]
    return out


def mark_literal(text, la, predicate, value, labels):
    anchor = DT_ANCHOR.get(predicate)
    if anchor is None:
        return None
    m = re.search(anchor(value), text, re.IGNORECASE)
    s1 = valid_spans(text, la, labels)
    if m is None or not s1:
        return None
    vm = re.search(re.escape(value), text[m.start():m.end()])
    if vm is None:
        return None
    s2, e2 = m.start() + vm.start(), m.start() + vm.end()
    a, b = next(((x, y) for x, y in s1 if y <= s2 or e2 <= x), (None, None))
    if a is None:
        return None
    out = text
    for s, e, tag in sorted([(a, b, "E1"), (s2, e2, "E2")], key=lambda x: x[0], reverse=True):
        out = out[:s] + f"[{tag}]" + out[s:e] + f"[/{tag}]" + out[e:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-datatype", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    corpus = load_jsonl(GEN / "story_corpus.jsonl")
    gold = load_jsonl(GEN / "gold_triples.jsonl")
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
    label_of.update(SURFACE_OVERRIDES)
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

    misses = 0
    for story in corpus:
        split = story["split"]
        for line in story["lines"]:
            txt = line["text"]
            for tr in line["triples"]:
                s, p = tr["subject"], tr["predicate"]
                if tr["object"] == "LiteralValue":
                    if args.no_datatype:
                        continue
                    v = tr.get("literal")
                    if v is None:
                        continue
                    valid = mark_literal(txt, label_of[s], p, str(v), all_labels)
                    if valid is None:
                        continue
                    add(valid, "VALID", p, s, "LiteralValue", None, "datatype", split)
                    pool = [q for q in data_props if q != p and compat_dom(q, s)]
                    if pool:
                        add(valid, "INVALID", rng.choice(pool), s, "LiteralValue", "hard", "datatype", split)
                    continue
                o = tr["object"]
                ls, lo = label_of[s], label_of[o]
                valid = mark_pair(txt, ls, lo, all_labels)
                if valid is None:
                    misses += 1
                    continue
                add(valid, "VALID", p, s, o, None, "object", split)
                hard = [q for q in obj_props if q != p and compat(q, s, o) and q not in pair_preds[(s, o)]]
                if hard:
                    add(valid, "INVALID", rng.choice(hard), s, o, "hard", "object", split)
                if (o, p, s) not in gold_obj:
                    rev = mark_pair(txt, lo, ls, all_labels)
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
        "source": "story_corpus.jsonl (300 stories)",
        "total_rows": len(rows),
        "label_distribution": dict(Counter(r["label"] for r in rows)),
        "neg_type_distribution": dict(Counter(r["neg_type"] for r in rows if r["neg_type"])),
        "rel_kind_distribution": dict(Counter(r["rel_kind"] for r in rows)),
        "split_distribution": {k: len(v) for k, v in by_split.items()},
        "split_label": {k: dict(Counter(r["label"] for r in v)) for k, v in by_split.items()},
        "distinct_candidate_relations": len(set(r["candidate_relation"] for r in rows)),
        "object_line_marking_misses": misses,
        "train_rows_purged_for_val_overlap": purged,
    }
    (GEN / "verifier_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
