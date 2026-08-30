"""
Pipeline 1 (Architecture 1) - Phase 0..2 for the Caves of Qud ontology.

What this script does (the "dynamic / ontology-driven" half of the pipeline):
  0. Parse the OWL ontology (classes, object/datatype properties with domain/range
     incl. owl:unionOf, inverseOf, individuals with their types + ABox assertions).
  1. Build `ontology_interface.json`  -> alias dictionary, relation vocabulary,
     class hierarchy, inverse map (the runtime "schema" the verifier is decoupled from).
  2a. Extract GOLD triples directly from the populated ABox -> `gold_triples.jsonl`
      (this ontology is already populated, so gold supervision is free).
  2b. Mention detection over a text corpus + ontology-constrained candidate
      generation (exhaustive pairwise, filtered by domain/range) -> candidates.

By default the text corpus is the rdfs:comment of every individual (a self-contained
smoke test). Point --input-dir at real .txt files later.

Usage:
  python build_qud_dataset.py
  python build_qud_dataset.py --input-dir ../data-input --glob "*.txt"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from rdflib.collection import Collection


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OWL = BASE_DIR.parent / "Final_Caves_of_QUD.owl"
DEFAULT_OUT = BASE_DIR / "generated"
NS = "http://www.semanticweb.org/chris/ontologies/caverne-di-qud#"


def short(uri: str) -> str:
    return uri.split("#", 1)[1] if "#" in uri else uri.rsplit("/", 1)[-1]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def label_to_text(label: str) -> str:
    """Human surface form: underscores -> spaces, collapse whitespace."""
    return normalize_space(label.replace("_", " "))


def build_alias_pattern(alias: str) -> re.Pattern[str]:
    """Word-boundary, case-insensitive, multi-word phrase match."""
    words = [re.escape(w) for w in normalize_space(alias).split()]
    body = r"\s+".join(words)
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", flags=re.IGNORECASE)


ROLE_SURFACE_ALIASES = {
    "Fighter_Boss": ("boss-tier fighter",),
    "Fighter_NonBoss": ("non-boss fighter",),
    "Trader_high": ("high-tier merchant",),
    "Trader_mid": ("mid-tier merchant",),
    "Trader_low": ("low-tier merchant",),
}


def resolve_class_expr(graph: Graph, node) -> set[str]:
    """A domain/range node may be a class URI or a blank node with owl:unionOf."""
    if isinstance(node, URIRef):
        return {str(node)}
    out: set[str] = set()
    union = graph.value(node, OWL.unionOf)
    if union is not None:
        for member in Collection(graph, union):
            if isinstance(member, URIRef):
                out.add(str(member))
    return out


def parse_ontology(owl_path: Path) -> dict:
    g = Graph()
    g.parse(owl_path)

    classes = {str(s) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    obj_props = {str(s) for s in g.subjects(RDF.type, OWL.ObjectProperty)}
    data_props = {str(s) for s in g.subjects(RDF.type, OWL.DatatypeProperty)}
    individuals = {str(s) for s in g.subjects(RDF.type, OWL.NamedIndividual)}

    def lbl(uri: str) -> str:
        v = g.value(URIRef(uri), RDFS.label)
        return str(v) if v is not None else label_to_text(short(uri))

    direct_parents: dict[str, set[str]] = defaultdict(set)
    for c, _, p in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(c, URIRef) and isinstance(p, URIRef):
            direct_parents[str(c)].add(str(p))

    def ancestors(cls: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        for parent in direct_parents.get(cls, ()):
            if parent not in seen:
                seen.add(parent)
                ancestors(parent, seen)
        return seen

    class_ancestors = {c: ({c} | ancestors(c)) for c in classes}

    prop_domain: dict[str, set[str]] = {}
    prop_range: dict[str, set[str]] = {}
    inverse_of: dict[str, str] = {}
    for p in obj_props | data_props:
        pu = URIRef(p)
        dom: set[str] = set()
        for d in g.objects(pu, RDFS.domain):
            dom |= resolve_class_expr(g, d)
        rng: set[str] = set()
        for r in g.objects(pu, RDFS.range):
            rng |= resolve_class_expr(g, r)
        prop_domain[p] = dom
        prop_range[p] = rng
        inv = g.value(pu, OWL.inverseOf)
        if isinstance(inv, URIRef):
            inverse_of[p] = str(inv)
            inverse_of[str(inv)] = p

    ind_types: dict[str, list[str]] = {}
    for ind in individuals:
        types = [str(t) for t in g.objects(URIRef(ind), RDF.type)
                 if isinstance(t, URIRef) and str(t) in classes]
        ind_types[ind] = types

    aliases: list[dict] = []
    for ind in individuals:
        forms = {lbl(ind), label_to_text(short(ind))}
        forms |= set(ROLE_SURFACE_ALIASES.get(short(ind), ()))
        for form in forms:
            aliases.append({"alias": form, "uri": ind, "kind": "individual",
                            "short": short(ind), "types": ind_types[ind]})
    for cls in classes:
        forms = {lbl(cls), label_to_text(short(cls))}
        for form in forms:
            aliases.append({"alias": form, "uri": cls, "kind": "class",
                            "short": short(cls), "types": [cls]})
    aliases.sort(key=lambda a: len(a["alias"]), reverse=True)

    relation_vocabulary = [
        {"uri": p, "short": short(p), "label": lbl(p),
         "kind": "object_property" if p in obj_props else "datatype_property",
         "domain": sorted(short(d) for d in prop_domain[p]),
         "range": sorted(short(r) for r in prop_range[p]),
         "inverse": short(inverse_of[p]) if p in inverse_of else None}
        for p in sorted(obj_props | data_props, key=short)
    ]

    return {
        "graph": g,
        "classes": classes,
        "obj_props": obj_props,
        "data_props": data_props,
        "individuals": individuals,
        "ind_types": ind_types,
        "class_ancestors": class_ancestors,
        "prop_domain": prop_domain,
        "prop_range": prop_range,
        "inverse_of": inverse_of,
        "aliases": aliases,
        "relation_vocabulary": relation_vocabulary,
        "label_of": {uri: lbl(uri) for uri in individuals | classes | obj_props | data_props},
    }


def extract_gold_triples(onto: dict) -> list[dict]:
    g: Graph = onto["graph"]
    rows: list[dict] = []
    for ind in sorted(onto["individuals"], key=short):
        su = URIRef(ind)
        for p, o in g.predicate_objects(su):
            ps = str(p)
            if ps in onto["obj_props"] and isinstance(o, URIRef) and str(o) in onto["individuals"]:
                rows.append({
                    "subject": short(ind), "predicate": short(ps), "object": short(str(o)),
                    "subject_label": onto["label_of"].get(ind, label_to_text(short(ind))),
                    "object_label": onto["label_of"].get(str(o), label_to_text(short(str(o)))),
                    "subject_types": [short(t) for t in onto["ind_types"][ind]],
                    "object_types": [short(t) for t in onto["ind_types"].get(str(o), [])],
                    "kind": "object_property",
                })
            elif ps in onto["data_props"] and isinstance(o, Literal):
                rows.append({
                    "subject": short(ind), "predicate": short(ps), "object": str(o),
                    "subject_label": onto["label_of"].get(ind, label_to_text(short(ind))),
                    "object_label": str(o),
                    "subject_types": [short(t) for t in onto["ind_types"][ind]],
                    "object_types": ["LiteralValue"],
                    "datatype": short(str(o.datatype)) if o.datatype else "string",
                    "kind": "datatype_property",
                })
    return rows


def detect_mentions(text: str, onto: dict) -> list[dict]:
    occupied: list[tuple[int, int]] = []
    mentions: list[dict] = []
    for entry in onto["aliases"]:
        for m in build_alias_pattern(entry["alias"]).finditer(text):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or e <= span[0]) for s, e in occupied):
                continue
            occupied.append(span)
            mentions.append({"uri": entry["uri"], "short": entry["short"], "kind": entry["kind"],
                             "types": entry["types"], "matched_text": text[span[0]:span[1]],
                             "start": span[0], "end": span[1]})
    mentions.sort(key=lambda x: (x["start"], x["end"]))
    return mentions


def type_set(types: list[str], onto: dict) -> set[str]:
    """Expand an individual's asserted types to include all superclasses."""
    out: set[str] = set()
    for t in types:
        out |= onto["class_ancestors"].get(t, {t})
    return out


def compatible(prop: str, e1: dict, e2: dict, onto: dict) -> bool:
    dom = onto["prop_domain"].get(prop, set())
    rng = onto["prop_range"].get(prop, set())
    s1 = type_set(e1["uri_types"], onto)
    s2 = type_set(e2["uri_types"], onto)
    ok_dom = (not dom) or bool(dom & s1)
    ok_rng = (not rng) or bool(rng & s2)
    return ok_dom and ok_rng


def mark(text: str, e1: dict, e2: dict) -> str:
    spans = sorted([(e1["start"], e1["end"], "E1"), (e2["start"], e2["end"], "E2")],
                   key=lambda x: x[0], reverse=True)
    out = text
    for s, e, tag in spans:
        out = out[:s] + f"[{tag}]" + out[s:e] + f"[/{tag}]" + out[e:]
    return out


def generate_candidates(text: str, onto: dict, max_pairs: int = 60) -> tuple[list[dict], list[dict]]:
    mentions = detect_mentions(text, onto)
    for mtn in mentions:
        if mtn["kind"] == "individual":
            mtn["uri_types"] = onto["ind_types"].get(mtn["uri"], [])
        else:
            mtn["uri_types"] = [mtn["uri"]]
    inds = [m for m in mentions if m["kind"] == "individual"]
    candidates: list[dict] = []
    pairs = 0
    for i, e1 in enumerate(inds):
        for j, e2 in enumerate(inds):
            if i == j:
                continue
            pairs += 1
            if pairs > max_pairs:
                break
            for prop in sorted(onto["obj_props"], key=short):
                if compatible(prop, e1, e2, onto):
                    candidates.append({
                        "candidate_relation": short(prop),
                        "subject": e1["short"], "object": e2["short"],
                        "text": f"[REL] {short(prop)} [/REL] " + mark(text, e1, e2),
                        "sentence": text,
                    })
    return mentions, candidates


def corpus_from_comments(onto: dict) -> list[dict]:
    g: Graph = onto["graph"]
    rows: list[dict] = []
    for ind in sorted(onto["individuals"], key=short):
        c = g.value(URIRef(ind), RDFS.comment)
        if c:
            rows.append({"sentence_id": short(ind), "source": "rdfs:comment",
                         "text": normalize_space(str(c))})
    return rows


def corpus_from_files(input_dir: Path, glob: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob(glob)):
        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, chunk in enumerate(re.split(r"(?<=[.!?])\s+|\n+", text), start=1):
            s = normalize_space(chunk)
            if s:
                rows.append({"sentence_id": f"{path.stem}_s{idx:03d}", "source": str(path), "text": s})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", type=Path, default=DEFAULT_OWL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--glob", type=str, default="*.txt")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    onto = parse_ontology(args.owl)

    interface = {
        "ontology": str(args.owl),
        "classes": sorted(short(c) for c in onto["classes"]),
        "relation_vocabulary": onto["relation_vocabulary"],
        "class_ancestors": {
            short(c): sorted(short(a) for a in anc)
            for c, anc in onto["class_ancestors"].items()
        },
        "individual_types": {
            short(i): [short(t) for t in onto["ind_types"][i]]
            for i in onto["individuals"]
        },
        "alias_dictionary": [
            {"alias": a["alias"], "short": a["short"], "kind": a["kind"]}
            for a in onto["aliases"]
        ],
    }
    (args.out / "ontology_interface.json").write_text(
        json.dumps(interface, indent=2, ensure_ascii=False), encoding="utf-8")

    gold = extract_gold_triples(onto)
    write_jsonl(args.out / "gold_triples.jsonl", gold)

    corpus = (corpus_from_files(args.input_dir, args.glob)
              if args.input_dir else corpus_from_comments(onto))
    sent_rows: list[dict] = []
    cand_rows: list[dict] = []
    for row in corpus:
        mentions, candidates = generate_candidates(row["text"], onto)
        sent_rows.append({**row, "mentions": [
            {"short": m["short"], "kind": m["kind"], "matched_text": m["matched_text"]}
            for m in mentions]})
        for c in candidates:
            cand_rows.append({**row, **c})
    write_jsonl(args.out / "sentences_all.jsonl", sent_rows)
    write_jsonl(args.out / "relation_candidates_all.jsonl", cand_rows)

    gold_by_pred: dict[str, int] = defaultdict(int)
    for t in gold:
        gold_by_pred[t["predicate"]] += 1
    stats = {
        "classes": len(onto["classes"]),
        "object_properties": len(onto["obj_props"]),
        "datatype_properties": len(onto["data_props"]),
        "individuals": len(onto["individuals"]),
        "gold_triples_total": len(gold),
        "gold_triples_object": sum(1 for t in gold if t["kind"] == "object_property"),
        "gold_triples_datatype": sum(1 for t in gold if t["kind"] == "datatype_property"),
        "gold_triples_by_predicate": dict(sorted(gold_by_pred.items(), key=lambda x: -x[1])),
        "corpus_sentences": len(corpus),
        "sentences_with_mentions": sum(1 for s in sent_rows if s["mentions"]),
        "total_candidates": len(cand_rows),
    }
    (args.out / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nWrote ->", args.out)


if __name__ == "__main__":
    main()
