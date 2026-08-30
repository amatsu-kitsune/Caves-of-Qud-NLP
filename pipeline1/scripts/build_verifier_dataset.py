"""
Pipeline 1 (Architecture 1) - Phase 3+4 for Caves of Qud.

Turns gold ABox triples + LLM-grounded sentences into the SUPERVISED VERIFIER
dataset: one candidate per row, `text` carrying [REL]..[/REL] [E1]..[/E1] [E2]..[/E2],
`label` in {VALID, INVALID}. This is the schema-agnostic signal that lets the
verifier survive ontology changes without retraining.

Negatives per VALID (the core of Architecture 1):
  - hard      : a DIFFERENT predicate that is type-compatible (domain/range ok) for
                the same (E1,E2) but is not asserted -> teaches fine distinctions.
  - direction : the SAME predicate with E1/E2 swapped -> teaches argument order.
  - dr        : a predicate whose domain/range is violated -> easy negative.

Inputs  (from generated/):  gold_triples.jsonl, gold_sentences.jsonl, ontology_interface.json
Outputs (to generated/):    verifier_{train,val,test,all}.jsonl, verifier_stats.json

Usage:
  python build_verifier_dataset.py
  python build_verifier_dataset.py --include-datatype --seed 13
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def find_ci(haystack: str, needle: str) -> int:
    return haystack.lower().find(needle.lower())


def all_spans(text: str, label: str) -> list[tuple[int, int]]:
    pat = r"(?<!\w)" + re.escape(label) + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pat, text, flags=re.IGNORECASE)]


def valid_spans(text: str, label: str, labels: list[str] | None) -> list[tuple[int, int]]:
    """Spans of `label` NOT nested inside a longer entity mention.
    Prevents tagging the 'Omonporch' that is part of 'Asphodel Earl of Omonporch'."""
    raw = all_spans(text, label)
    if not labels:
        return raw
    own = len(label)
    longer: list[tuple[int, int]] = []
    for other in labels:
        if len(other) > own:
            longer += all_spans(text, other)
    return [(s, e) for (s, e) in raw
            if not any(ls <= s and e <= le and (ls < s or e < le) for ls, le in longer)]


def mark_pair(text: str, label_e1: str, label_e2: str, labels: list[str] | None = None) -> str | None:
    """Wrap E1/E2 with markers, choosing NON-overlapping, non-nested occurrences."""
    spans1 = valid_spans(text, label_e1, labels)
    spans2 = valid_spans(text, label_e2, labels)
    if not spans1 or not spans2:
        return None
    best = None
    for s1, e1 in spans1:
        for s2, e2 in spans2:
            if e1 <= s2 or e2 <= s1:
                span = max(e1, e2)
                if best is None or span < best[0]:
                    best = (span, (s1, e1), (s2, e2))
    if best is None:
        return None
    (s1, e1), (s2, e2) = best[1], best[2]
    spans = sorted([(s1, e1, "E1"), (s2, e2, "E2")], key=lambda x: x[0], reverse=True)
    out = text
    for s, e, tag in spans:
        out = out[:s] + f"[{tag}]" + out[s:e] + f"[/{tag}]" + out[e:]
    return out


DT_ANCHOR = {
    "HP": lambda v: rf"\b{re.escape(v)}\s+HP\b",
    "level": lambda v: rf"\blevel\s+{re.escape(v)}\b",
    "strata": lambda v: rf"\bstrata\s+{re.escape(v)}\b",
    "Value": lambda v: rf"\b{re.escape(v)}\s+drams\b",
    "Weight": lambda v: rf"\bweight\s+is\s+{re.escape(v)}\b",
}


def mark_literal(text: str, label_e1: str, predicate: str, value: str) -> str | None:
    anchor = DT_ANCHOR.get(predicate)
    if anchor is None:
        return None
    m = re.search(anchor(value), text, flags=re.IGNORECASE)
    i1 = find_ci(text, label_e1)
    if m is None or i1 < 0:
        return None
    vm = re.search(re.escape(value), text[m.start():m.end()])
    if vm is None:
        return None
    s2, e2 = m.start() + vm.start(), m.start() + vm.end()
    s1, e1 = i1, i1 + len(label_e1)
    if not (e1 <= s2 or e2 <= s1):
        return None
    spans = sorted([(s1, e1, "E1"), (s2, e2, "E2")], key=lambda x: x[0], reverse=True)
    out = text
    for s, e, tag in spans:
        out = out[:s] + f"[{tag}]" + out[s:e] + f"[/{tag}]" + out[e:]
    return out


def split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-datatype", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = load_jsonl(GEN / "gold_triples.jsonl")
    sentences = load_jsonl(GEN / "gold_sentences.jsonl")
    iface = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))

    texts = [s["text"] for s in sentences]
    relmeta = {r["short"]: r for r in iface["relation_vocabulary"]}
    obj_props = [r["short"] for r in iface["relation_vocabulary"] if r["kind"] == "object_property"]
    data_props = [r["short"] for r in iface["relation_vocabulary"] if r["kind"] == "datatype_property"]
    ancestors = iface["class_ancestors"]
    ind_types = iface["individual_types"]

    label_of: dict[str, str] = {}
    for t in gold:
        label_of[t["subject"]] = t["subject_label"]
        if t["kind"] == "object_property":
            label_of[t["object"]] = t["object_label"]

    SURFACE_OVERRIDES = {
        "Fighter_Boss": "boss-tier Fighter",
        "Fighter_NonBoss": "non-boss Fighter",
        "Trader_high": "high-tier merchant",
        "Trader_mid": "mid-tier merchant",
        "Trader_low": "low-tier merchant",
    }
    label_of.update(SURFACE_OVERRIDES)

    def closure(ind: str) -> set[str]:
        out: set[str] = set()
        for ty in ind_types.get(ind, []):
            out |= set(ancestors.get(ty, [ty]))
        return out

    def compat(pred: str, s: str, o: str) -> bool:
        dom = set(relmeta[pred]["domain"])
        rng = set(relmeta[pred]["range"])
        cs, co = closure(s), closure(o)
        return ((not dom) or bool(dom & cs)) and ((not rng) or bool(rng & co))

    def compat_domain(pred: str, s: str) -> bool:
        dom = set(relmeta[pred]["domain"])
        return (not dom) or bool(dom & closure(s))

    gold_obj = {(t["subject"], t["predicate"], t["object"]) for t in gold if t["kind"] == "object_property"}
    pair_preds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s, p, o in gold_obj:
        pair_preds[(s, o)].add(p)

    rows: list[dict] = []
    misses = 0

    def add(text_marked: str, label: str, pred: str, s: str, o: str, neg: str | None, kind: str, split: str):
        rows.append({
            "text": f"[REL] {pred} [/REL] {text_marked}",
            "label": label, "candidate_relation": pred,
            "subject": s, "object": o, "neg_type": neg, "rel_kind": kind,
            "split": split,
        })

    all_labels = sorted(set(label_of.values()), key=len, reverse=True)
    story_by_id = {s["subject_id"]: s["text"] for s in sentences}

    def find_text(la: str, lb: str, prefer: str | None = None) -> str | None:
        order = ([prefer] if prefer else []) + [t for t in texts if t != prefer]
        for txt in order:
            if mark_pair(txt, la, lb, all_labels) is not None:
                return txt
        return None

    for t in gold:
        if t["kind"] != "object_property":
            continue
        s, p, o = t["subject"], t["predicate"], t["object"]
        ls, lo = label_of[s], label_of[o]
        txt = find_text(ls, lo, story_by_id.get(s))
        if txt is None:
            misses += 1
            continue
        valid = mark_pair(txt, ls, lo, all_labels)
        if valid is None:
            misses += 1
            continue
        sp = split_of(txt)
        add(valid, "VALID", p, s, o, None, "object", sp)

        hard_pool = [q for q in obj_props
                     if q != p and compat(q, s, o) and q not in pair_preds[(s, o)]]
        if hard_pool:
            add(valid, "INVALID", rng_choice(rng, hard_pool), s, o, "hard", "object", sp)

        if (o, p, s) not in gold_obj:
            rev = mark_pair(txt, lo, ls, all_labels)
            if rev is not None:
                add(rev, "INVALID", p, o, s, "direction", "object", sp)
        else:
            dr_pool = [q for q in obj_props if not compat(q, s, o)]
            if dr_pool:
                add(valid, "INVALID", rng_choice(rng, dr_pool), s, o, "dr", "object", sp)

    if args.include_datatype:
        for t in gold:
            if t["kind"] != "datatype_property":
                continue
            s, p, v = t["subject"], t["predicate"], t["object"]
            ls = label_of[s]
            txt = next((x for x in texts if find_ci(x, ls) >= 0
                        and re.search(DT_ANCHOR.get(p, lambda _: r"$^")(v), x, re.I)), None)
            if txt is None:
                continue
            valid = mark_literal(txt, ls, p, v)
            if valid is None:
                continue
            sp = split_of(txt)
            add(valid, "VALID", p, s, "LiteralValue", None, "datatype", sp)
            hard_pool = [q for q in data_props if q != p and compat_domain(q, s)]
            if hard_pool:
                hard = mark_literal(txt, ls, p, v)
                if hard is not None:
                    add(hard, "INVALID", rng_choice(rng, hard_pool), s, "LiteralValue", "hard", "datatype", sp)

    by_split: dict[str, list[dict]] = defaultdict(list)
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
        "total_rows": len(rows),
        "label_distribution": dict(Counter(r["label"] for r in rows)),
        "neg_type_distribution": dict(Counter(r["neg_type"] for r in rows if r["neg_type"])),
        "rel_kind_distribution": dict(Counter(r["rel_kind"] for r in rows)),
        "split_distribution": {k: len(v) for k, v in by_split.items()},
        "split_label": {k: dict(Counter(r["label"] for r in v)) for k, v in by_split.items()},
        "gold_object_triples": len(gold_obj),
        "object_triples_unmatched_to_text": misses,
    }
    (GEN / "verifier_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def rng_choice(rng: random.Random, seq: list):
    return seq[rng.randrange(len(seq))]


if __name__ == "__main__":
    main()
