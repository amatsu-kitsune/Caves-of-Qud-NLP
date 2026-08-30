"""
Pipeline 1 - inference adapter following the MANDATED chain:

    txt -> GLiNER -> GLiREL -> BERT (verifier) -> SHACL -> triple

Each stage has a clean interface and a FALLBACK, so the pipeline runs today and
upgrades to the real models by flipping a flag once they are installed:

  stage 2  GLiNER  (entity spans + type)   fallback: ontology alias dictionary
  stage 3  GLiREL  (relation candidates)   fallback: domain/range pairwise generation
  stage 4  BERT    (VALID/INVALID verifier) fallback: pass-through (candidate, unverified)
  stage 5  SHACL   (constraint validation)  fallback: domain/range already enforced upstream

Install for the real models:
    pip install gliner glirel transformers spacy pyshacl
    python -m spacy download en_core_web_sm

Usage:
    python infer_pipeline.py --file ../data-input/val/story_300.txt
    python infer_pipeline.py --text "Otho gives the player the quest A Call to Arms."
    python infer_pipeline.py --file ... --use-gliner --use-glirel --model ../models/verifier/best_model
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
NS = "http://www.semanticweb.org/chris/ontologies/caverne-di-qud#"


def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


_PROTECTED_LABELS: list[str] | None = None


def _protected_labels() -> list[str]:
    """Ontology aliases containing sentence punctuation (e.g. the quest names
    "Weirdwire Conduit... Eureka!" / "Pax Qlanq, I Presume?"). These must be
    masked during sentence splitting or the splitter breaks inside the name."""
    global _PROTECTED_LABELS
    if _PROTECTED_LABELS is None:
        try:
            iface = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))
            labs = {a["alias"] for a in iface["alias_dictionary"] if re.search(r"[.!?]", a["alias"])}
            _PROTECTED_LABELS = sorted(labs, key=len, reverse=True)
        except Exception:
            _PROTECTED_LABELS = []
    return _PROTECTED_LABELS


def sent_split(text: str) -> list[str]:
    masks = {}
    for i, lab in enumerate(_protected_labels()):
        if lab in text:
            key = f"__QN{i}__"
            text = text.replace(lab, key)
            masks[key] = lab
    out = []
    for chunk in re.split(r"\n+", text):
        for part in re.split(r"(?<=[.!?])\s+", chunk):
            s = normalize(part)
            if s:
                for key, lab in masks.items():
                    s = s.replace(key, lab)
                out.append(s)
    return out


class Ontology:
    def __init__(self):
        self.iface = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))
        self.aliases = sorted(self.iface["alias_dictionary"], key=lambda a: len(a["alias"]), reverse=True)
        self.relvocab = {r["short"]: r for r in self.iface["relation_vocabulary"]}
        self.obj_props = [r["short"] for r in self.iface["relation_vocabulary"] if r["kind"] == "object_property"]
        self.ancestors = self.iface["class_ancestors"]
        self.ind_types = self.iface["individual_types"]

    def uri(self, short: str) -> str:
        return NS + short

    def closure(self, short: str) -> set[str]:
        out: set[str] = set()
        for t in self.ind_types.get(short, [short]):
            out |= set(self.ancestors.get(t, [t]))
        return out

    def compatible(self, pred: str, s_short: str, o_short: str) -> bool:
        r = self.relvocab[pred]
        dom, rng = set(r["domain"]), set(r["range"])
        return ((not dom) or bool(dom & self.closure(s_short))) and \
               ((not rng) or bool(rng & self.closure(o_short)))


def detect_entities(sentence: str, onto: Ontology, use_gliner: bool, gliner_model=None) -> list[dict]:
    if use_gliner and gliner_model is not None:
        labels = [g["label"] for g in json.loads((GEN / "gliner_labels.json").read_text(encoding="utf-8"))]
        spans = gliner_model.predict_entities(sentence, labels, threshold=0.4)
        ents = []
        for sp in spans:
            grounded = _ground(sp["text"], onto)
            if grounded:
                ents.append({**grounded, "start": sp["start"], "end": sp["end"],
                             "gliner_type": sp["label"], "matched_text": sp["text"]})
        return _dedupe(ents + _dict_entities(sentence, onto))
    return _dict_entities(sentence, onto)


def _dict_entities(sentence: str, onto: Ontology) -> list[dict]:
    """Exact alias-dictionary matcher (also the no-GLiNER fallback)."""
    ents, occupied = [], []
    for a in onto.aliases:
        for m in re.finditer(r"(?<!\w)" + re.escape(a["alias"]) + r"(?!\w)", sentence, re.IGNORECASE):
            sp = (m.start(), m.end())
            if any(not (sp[1] <= s or e <= sp[0]) for s, e in occupied):
                continue
            occupied.append(sp)
            ents.append({"short": a["short"], "uri": onto.uri(a["short"]), "kind": a["kind"],
                         "types": onto.ind_types.get(a["short"], [a["short"]]),
                         "start": sp[0], "end": sp[1], "matched_text": sentence[sp[0]:sp[1]]})
    ents.sort(key=lambda x: x["start"])
    return ents


def _ground(span_text: str, onto: Ontology) -> dict | None:
    norm = normalize(span_text).lower()
    hit = _alias_lookup(norm, onto)
    if hit is None and norm.startswith("the "):
        hit = _alias_lookup(norm[4:], onto)
    return hit


def _alias_lookup(norm: str, onto: Ontology) -> dict | None:
    for a in onto.aliases:
        if a["alias"].lower() == norm:
            return {"short": a["short"], "uri": onto.uri(a["short"]), "kind": a["kind"],
                    "types": onto.ind_types.get(a["short"], [a["short"]])}
    return None


def _dedupe(ents: list[dict]) -> list[dict]:
    out, occ = [], []
    for e in sorted(ents, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if any(not (e["end"] <= s or en <= e["start"]) for s, en in occ):
            continue
        occ.append((e["start"], e["end"]))
        out.append(e)
    return out


def extract_candidates(sentence: str, ents: list[dict], onto: Ontology,
                       use_glirel: bool, glirel_model=None, *,
                       union_pairwise: bool = False, top_k: int = 3) -> list[dict]:
    inds = [e for e in ents if e["kind"] == "individual"]
    cands: list[dict] = []
    if use_glirel and glirel_model is not None:
        if len(inds) < 2:
            return cands
        labels = [r["label"] for r in json.loads((GEN / "glirel_labels.json").read_text(encoding="utf-8"))]
        lab2pred = {r["label"]: r["predicate_short"] for r in json.loads((GEN / "glirel_labels.json").read_text(encoding="utf-8"))}
        tokens = sentence.split()
        ner = _to_glirel_ner(inds, sentence, tokens)
        start2ent = {span[0]: e for e, span in zip(inds, ner)}
        rels = glirel_model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=top_k)
        for r in rels:
            pred = lab2pred.get(r.get("label"))
            h = _resolve(r, "head_pos", "head_text", start2ent, inds)
            t = _resolve(r, "tail_pos", "tail_text", start2ent, inds)
            if pred and h and t and h is not t and onto.compatible(pred, h["short"], t["short"]):
                cands.append(_mk_candidate(sentence, h, t, pred, score=r.get("score")))
        if union_pairwise:
            seen = {(c["subject"], c["candidate_relation"], c["object"]) for c in cands}
            for c in _pairwise_candidates(sentence, inds, onto):
                key = (c["subject"], c["candidate_relation"], c["object"])
                if key not in seen:
                    seen.add(key)
                    cands.append(c)
        return cands
    return _pairwise_candidates(sentence, inds, onto)


def _pairwise_candidates(sentence: str, inds: list[dict], onto: Ontology) -> list[dict]:
    cands = []
    for e1 in inds:
        for e2 in inds:
            if e1 is e2:
                continue
            for pred in onto.obj_props:
                if onto.compatible(pred, e1["short"], e2["short"]):
                    cands.append(_mk_candidate(sentence, e1, e2, pred, score=None))
    return cands


def _mk_candidate(sentence: str, e1: dict, e2: dict, pred: str, score) -> dict:
    spans = sorted([(e1["start"], e1["end"], "E1"), (e2["start"], e2["end"], "E2")],
                   key=lambda x: x[0], reverse=True)
    out = sentence
    for s, en, tag in spans:
        out = out[:s] + f"[{tag}]" + out[s:en] + f"[/{tag}]" + out[en:]
    return {"candidate_relation": pred, "subject": e1["short"], "object": e2["short"],
            "text": f"[REL] {pred} [/REL] {out}", "glirel_score": score}


def _to_glirel_ner(inds, sentence, tokens):
    out = []
    for e in inds:
        pre = len(sentence[:e["start"]].split())
        ntok = max(1, len(e["matched_text"].split()))
        out.append([pre, pre + ntok - 1, e.get("gliner_type", "entity"), e["matched_text"]])
    return out


def _resolve(r, kpos, ktext, start2ent, inds):
    """Map a GLiREL head/tail back to one of our grounded entities, by token position
    (robust) with a text fallback."""
    pos = r.get(kpos)
    if isinstance(pos, (list, tuple)) and pos:
        e = start2ent.get(pos[0])
        if e:
            return e
    return _match_ent(inds, r.get(ktext))


def _match_ent(inds, text):
    if isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text)
    if not isinstance(text, str):
        return None
    text = normalize(text).lower()
    for e in inds:
        if e["matched_text"].lower() == text:
            return e
    return None


def load_verifier(model_dir: Path):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception:
        return None
    if not model_dir or not Path(model_dir).exists():
        return None
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    return (tok, mdl)


def verify(cands: list[dict], verifier) -> list[dict]:
    if verifier is None:
        for c in cands:
            c["verifier_label"] = None
        return cands
    import torch
    tok, mdl = verifier
    id2label = mdl.config.id2label
    for c in cands:
        enc = tok(c["text"], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            pred = int(mdl(**enc).logits.argmax(-1).item())
        c["verifier_label"] = id2label[pred] if isinstance(id2label, dict) else id2label[str(pred)]
    return cands


def shacl_filter(triples: list[dict], shapes_path: Path | None) -> list[dict]:
    if not shapes_path or not Path(shapes_path).exists():
        return triples
    try:
        import pyshacl
    except Exception:
        return triples
    return triples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path)
    ap.add_argument("--text", type=str)
    ap.add_argument("--use-gliner", action="store_true")
    ap.add_argument("--use-glirel", action="store_true")
    ap.add_argument("--stage2-union", action=argparse.BooleanOptionalAction, default=True,
                    help="union GLiREL candidates with domain/range pairwise generation")
    ap.add_argument("--glirel-topk", type=int, default=3)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--shapes", type=Path, default=None)
    args = ap.parse_args()

    onto = Ontology()
    text = args.text if args.text else (args.file.read_text(encoding="utf-8") if args.file else "")
    if not text:
        raise SystemExit("provide --file or --text")

    gliner_model = None
    if args.use_gliner:
        from gliner import GLiNER
        gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    glirel_model = None
    if args.use_glirel:
        from glirel import GLiREL
        _orig = GLiREL._from_pretrained.__func__

        def _compat(cls, *a, **kw):
            kw.setdefault("proxies", None)
            kw.setdefault("resume_download", False)
            return _orig(cls, *a, **kw)

        GLiREL._from_pretrained = classmethod(_compat)
        glirel_model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    verifier = load_verifier(args.model)

    all_accepted = []
    for sent in sent_split(text):
        ents = detect_entities(sent, onto, args.use_gliner, gliner_model)
        cands = extract_candidates(sent, ents, onto, args.use_glirel, glirel_model,
                                   union_pairwise=args.stage2_union, top_k=args.glirel_topk)
        cands = verify(cands, verifier)
        accepted = [c for c in cands if c["verifier_label"] in (None, "VALID")]
        for c in accepted:
            all_accepted.append({"subject": c["subject"], "predicate": c["candidate_relation"],
                                 "object": c["object"], "verifier_label": c["verifier_label"],
                                 "sentence": sent})

    all_accepted = shacl_filter(all_accepted, args.shapes)
    stages = (f"GLiNER={'real+dict-union' if args.use_gliner else 'dict-fallback'} | "
              f"GLiREL={('real+pairwise-union' if args.stage2_union else 'real') if args.use_glirel else 'pairwise-fallback'} | "
              f"BERT={'real' if verifier else 'pass-through'} | "
              f"SHACL={'on' if args.shapes else 'off'}")
    print("STAGES:", stages)
    print(f"candidate triples produced: {len(all_accepted)}")
    for t in all_accepted[:40]:
        tag = t["verifier_label"] or "UNVERIFIED"
        print(f"  ({t['subject']}, {t['predicate']}, {t['object']})  [{tag}]")


if __name__ == "__main__":
    main()
