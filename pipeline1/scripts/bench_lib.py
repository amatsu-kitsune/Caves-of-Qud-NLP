"""
Benchmark library for Pipeline 1.

Stages benchmarked (the MANDATED chain):
    txt -> (1) GLiNER -> (2) GLiREL -> (3) BERT/spaCy verifier -> (4) SHACL -> (5) KG assembly

This module is the shared backbone for the benchmark scripts. It deliberately
REUSES the production stage functions from `infer_pipeline.py` so the benchmark
and the real pipeline can never drift apart:

    stage 1  infer_pipeline.detect_entities
    stage 2  infer_pipeline.extract_candidates
    stage 3  this module's verifier ADAPTERS (BERT dir / spaCy / pass-through)
    stage 4  infer_pipeline.shacl_filter (+ optional pyshacl here)
    stage 5  assemble_kg() -> rdflib Graph -> Turtle

What it provides
----------------
* load_units(corpus)         : evaluation units (text, gold object-triple set) with tier tags
* Adapter classes            : BertVerifier / SpacyVerifier / PassThrough  (stage 3)
* run_pipeline_on_text(...)  : full chain on one text -> predicted triples + diagnostics
* score_units(...)           : stage diagnostics + triple P/R/F1 over a corpus
* assemble_kg(...)           : stage-5 RDF graph + stats (+ optional SHACL)
* prf(), discover_models()   : helpers
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import infer_pipeline as ip

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
MODELS = BASE / "models"
OUT = BASE / "outputs"
NS = ip.NS

BERT_ZOO = [
    ("bigbird-roberta-base",   "google/bigbird-roberta-base"),
    ("distilbert-base-uncased", "distilbert-base-uncased"),
    ("xlnet-base-cased",       "xlnet-base-cased"),
    ("roberta-base",           "roberta-base"),
    ("t5-base",                "t5-base"),
]


def model_dir_for(hf_id: str) -> Path:
    return MODELS / f"verifier_{hf_id.split('/')[-1].replace('-', '_')}"


def discover_models() -> list[dict]:
    """Find every trained verifier on disk: BERT dirs (best_model/) + spaCy."""
    found = []
    for d in sorted(MODELS.glob("verifier_*")):
        if (d / "best_model" / "config.json").exists():
            found.append({"name": d.name.replace("verifier_", ""), "type": "bert",
                          "path": d / "best_model"})
        elif (d / "meta.json").exists() or (d / "config.cfg").exists():
            found.append({"name": d.name.replace("verifier_", ""), "type": "spacy",
                          "path": d})
    return found


class PassThrough:
    """No verifier (stage-3 OFF): accept every GLiREL candidate. Measures the
    GLiREL precision ceiling and the verifier's filtering headroom."""
    name, type = "passthrough", "none"

    def label_texts(self, texts):
        return ["VALID"] * len(texts)


class BertVerifier:
    type = "bert"

    def __init__(self, model_dir, name=None, max_len=128, batch=32):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        id2 = self.mdl.config.id2label
        self.id2label = {int(k): v for k, v in id2.items()} if isinstance(id2, dict) else dict(enumerate(id2))
        self.max_len, self.batch = max_len, batch
        self.name = name or Path(model_dir).parent.name.replace("verifier_", "")

    def label_texts(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", truncation=True,
                           padding=True, max_length=self.max_len).to(self.device)
            with self.torch.no_grad():
                pred = self.mdl(**enc).logits.argmax(-1).cpu().tolist()
            out += [self.id2label[p] for p in pred]
        return out


class SpacyVerifier:
    type = "spacy"

    def __init__(self, model_dir, name=None):
        import spacy
        self.nlp = spacy.load(str(model_dir))
        self.name = name or Path(model_dir).name.replace("verifier_", "")

    def label_texts(self, texts):
        out = []
        for doc in self.nlp.pipe(texts):
            out.append(max(doc.cats, key=doc.cats.get) if doc.cats else "INVALID")
        return out


def load_adapter(spec: dict):
    if spec["type"] == "bert":
        return BertVerifier(spec["path"], name=spec["name"])
    if spec["type"] == "spacy":
        return SpacyVerifier(spec["path"], name=spec["name"])
    return PassThrough()


def _jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def load_units(corpus: str) -> list[dict]:
    units = []
    tiered_split = {"tiered_train": "train", "tiered_val": "val",
                    "tiered_test": "test"}.get(corpus)
    if corpus == "tiered" or tiered_split:
        for ex in _jsonl(GEN / "qud_tiered_annotated.jsonl"):
            if tiered_split and ex.get("split") != tiered_split:
                continue
            gold = {(t["subject"], t["predicate"], t["object"])
                    for t in ex["triples"] if t["kind"] == "object_property"}
            units.append({"id": ex["id"], "tier": ex["tier"],
                          "text": ex["text"], "gold": gold})
    return units


def run_pipeline_on_text(text, onto, adapter, *, use_gliner=False, gliner_model=None,
                         use_glirel=False, glirel_model=None,
                         stage2_union=False, glirel_topk=3):
    pred, cand_set, ent_set = set(), set(), set()
    for sent in ip.sent_split(text):
        ents = ip.detect_entities(sent, onto, use_gliner, gliner_model)
        for e in ents:
            if e["kind"] == "individual":
                ent_set.add(e["short"])
        cands = ip.extract_candidates(sent, ents, onto, use_glirel, glirel_model,
                                      union_pairwise=stage2_union, top_k=glirel_topk)
        for c in cands:
            cand_set.add((c["subject"], c["candidate_relation"], c["object"]))
        if not cands:
            continue
        labels = adapter.label_texts([c["text"] for c in cands])
        for c, lab in zip(cands, labels):
            if lab == "VALID":
                pred.add((c["subject"], c["candidate_relation"], c["object"]))
    return {"pred": pred, "cands": cand_set, "ents": ent_set}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_units(units, onto, adapter, *, use_gliner=False, gliner_model=None,
                use_glirel=False, glirel_model=None, per_tier=True,
                collect_per_unit=False, stage2_union=False, glirel_topk=3):
    """Run the chain over every unit; aggregate stage diagnostics + triple P/R/F1
    (overall and per tier). Also returns the global predicted/gold KG triple sets.

    When ``collect_per_unit`` is set, also returns ``per_unit``: one metrics row per
    input text (per txt) with its triple P/R/F1, tp/fp/fn and exact-match flag."""
    agg = {}
    glob_pred, glob_gold = set(), set()
    per_unit = [] if collect_per_unit else None

    def bucket(t):
        return agg.setdefault(t, dict(tp=0, fp=0, fn=0, ent_hit=0, ent_tot=0,
                                      cand_hit=0, cand_tot=0, n_cands=0, units=0))

    t0 = time.time()
    for u in units:
        res = run_pipeline_on_text(text=u["text"], onto=onto, adapter=adapter,
                                   use_gliner=use_gliner, gliner_model=gliner_model,
                                   use_glirel=use_glirel, glirel_model=glirel_model,
                                   stage2_union=stage2_union, glirel_topk=glirel_topk)
        gold, pred, cands, ents = u["gold"], res["pred"], res["cands"], res["ents"]
        glob_pred |= pred
        glob_gold |= gold
        utp, ufp, ufn = len(pred & gold), len(pred - gold), len(gold - pred)
        endpoints = {x for (s, _, o) in gold for x in (s, o)}
        for tier in ({u["tier"], "ALL"} if per_tier else {"ALL"}):
            b = bucket(tier)
            b["units"] += 1
            b["tp"] += utp
            b["fp"] += ufp
            b["fn"] += ufn
            b["ent_tot"] += len(endpoints)
            b["ent_hit"] += len(endpoints & ents)
            b["cand_tot"] += len(gold)
            b["cand_hit"] += len(gold & cands)
            b["n_cands"] += len(cands)
        if collect_per_unit:
            up, ur, uf = prf(utp, ufp, ufn)
            per_unit.append({
                "id": u["id"], "tier": u["tier"],
                "n_gold": len(gold), "n_pred": len(pred),
                "tp": utp, "fp": ufp, "fn": ufn,
                "precision": round(up, 4), "recall": round(ur, 4), "f1": round(uf, 4),
                "exact_match": int(pred == gold),
                "entity_recall": round(len(endpoints & ents) / len(endpoints), 4) if endpoints else None,
                "n_candidates": len(cands),
            })
    elapsed = time.time() - t0

    rows = {}
    for tier, b in agg.items():
        p, r, f = prf(b["tp"], b["fp"], b["fn"])
        rows[tier] = {
            "units": b["units"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
            "entity_recall": round(b["ent_hit"] / b["ent_tot"], 4) if b["ent_tot"] else None,
            "candidate_recall_ceiling": round(b["cand_hit"] / b["cand_tot"], 4) if b["cand_tot"] else None,
            "avg_candidates_per_unit": round(b["n_cands"] / b["units"], 2) if b["units"] else 0,
        }
    return {"per_tier": rows, "seconds": round(elapsed, 2),
            "kg_pred": glob_pred, "kg_gold": glob_gold, "per_unit": per_unit}


def assemble_kg(triples, onto, *, out_ttl: Path | None = None, shapes: Path | None = None):
    from rdflib import Graph, Namespace, RDF, URIRef
    g = Graph()
    QUD = Namespace(NS)
    g.bind("qud", QUD)
    typed = set()
    for (s, p, o) in sorted(triples):
        g.add((URIRef(NS + s), URIRef(NS + p), URIRef(NS + o)))
        for ent in (s, o):
            if ent in typed:
                continue
            typed.add(ent)
            for ty in onto.ind_types.get(ent, []):
                g.add((URIRef(NS + ent), RDF.type, URIRef(NS + ty)))
    stats = {"n_triples": len(triples), "n_entities": len(typed),
             "n_graph_statements": len(g)}
    if out_ttl:
        out_ttl.parent.mkdir(parents=True, exist_ok=True)
        g.serialize(destination=str(out_ttl), format="turtle")
        stats["turtle"] = str(out_ttl)
    if shapes and Path(shapes).exists():
        try:
            import pyshacl
            onto_g = Graph().parse(str(BASE.parent / "Final_Caves_of_QUD.owl"))
            conforms, _, txt = pyshacl.validate(
                data_graph=g, shacl_graph=Graph().parse(str(shapes)),
                ont_graph=onto_g, inference="rdfs", advanced=True)
            stats["shacl_conforms"] = bool(conforms)
            stats["shacl_report"] = txt[:2000]
        except Exception as e:
            stats["shacl_error"] = repr(e)
    return g, stats


def kg_metrics(pred, gold):
    tp = len(pred & gold)
    p, r, f = prf(tp, len(pred - gold), len(gold - pred))
    return {"kg_precision": round(p, 4), "kg_recall": round(r, 4), "kg_f1": round(f, 4),
            "kg_tp": tp, "kg_fp": len(pred - gold), "kg_fn": len(gold - pred),
            "kg_pred_size": len(pred), "kg_gold_size": len(gold)}


def _short(uri):
    s = str(uri)
    return s.split("#", 1)[1] if s.startswith(NS) else None


def shacl_filter_triples(triples, onto, shapes_path, max_iter=6):
    import pyshacl
    from rdflib import Graph, URIRef
    shapes_g = Graph().parse(str(shapes_path))
    onto_g = Graph().parse(str(BASE.parent / "Final_Caves_of_QUD.owl"))
    Q = ("PREFIX sh: <http://www.w3.org/ns/shacl#> "
         "SELECT ?f ?p ?v WHERE { ?r a sh:ValidationResult ; sh:focusNode ?f ; "
         "sh:resultPath ?p ; sh:value ?v . }")
    kept, removed, conforms = set(triples), [], True
    for _ in range(max_iter):
        g, _ = assemble_kg(kept, onto)
        conforms, report_g, _ = pyshacl.validate(
            data_graph=g, shacl_graph=shapes_g, ont_graph=onto_g,
            inference="rdfs", advanced=True)
        if conforms:
            break
        offending = set()
        for f, p, v in report_g.query(Q):
            if isinstance(v, URIRef):
                t = (_short(f), _short(p), _short(v))
                if None not in t and t in kept:
                    offending.add(t)
        if not offending:
            break
        removed.extend(sorted(offending))
        kept -= offending
    return kept, removed, bool(conforms)


def write_eval_graphs(pred, gold, onto, out_dir, model):
    from rdflib import Graph, Namespace, URIRef, Literal, RDF
    out_dir.mkdir(parents=True, exist_ok=True)
    tp, fp, fn = pred & gold, pred - gold, gold - pred

    assemble_kg(pred, onto, out_ttl=out_dir / f"kg_{model}_predicted.ttl")
    assemble_kg(gold, onto, out_ttl=out_dir / "kg_expected.ttl")

    g = Graph()
    g.bind("qud", Namespace(NS))
    status = URIRef(NS + "evalStatus")
    i = 0
    for st, group in (("TP", sorted(tp)), ("FP", sorted(fp)), ("FN", sorted(fn))):
        for (s, p, o) in group:
            su, pu, ou = URIRef(NS + s), URIRef(NS + p), URIRef(NS + o)
            g.add((su, pu, ou))
            node = URIRef(NS + f"eval_{model}_{i}")
            g.add((node, RDF.type, RDF.Statement))
            g.add((node, RDF.subject, su))
            g.add((node, RDF.predicate, pu))
            g.add((node, RDF.object, ou))
            g.add((node, status, Literal(st)))
            i += 1
    g.serialize(destination=str(out_dir / f"kg_{model}_labeled.ttl"), format="turtle")

    with (out_dir / f"kg_{model}_diff.tsv").open("w", encoding="utf-8") as h:
        h.write("status\tsubject\tpredicate\tobject\n")
        for st, group in (("TP", sorted(tp)), ("FP", sorted(fp)), ("FN", sorted(fn))):
            for (s, p, o) in group:
                h.write(f"{st}\t{s}\t{p}\t{o}\n")
    return {"tp": len(tp), "fp": len(fp), "fn": len(fn)}
