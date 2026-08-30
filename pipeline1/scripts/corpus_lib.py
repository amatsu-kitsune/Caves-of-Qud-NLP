"""
Shared corpus-building primitives for the tiered benchmark corpus.

Span-marking helpers and label surface overrides used by
build_verifier_from_tiered.py to build the verifier dataset from the
annotated tiered corpus.
"""
from __future__ import annotations

import json
import re

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
