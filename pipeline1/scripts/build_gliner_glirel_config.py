"""Emit GLiNER + GLiREL configuration from the ontology (stage 2 & 3 schema)."""
from __future__ import annotations
import json, re
from pathlib import Path
from rdflib import Graph, URIRef, RDF, RDFS, OWL

BASE = Path(__file__).resolve().parents[1]
OWL_PATH = BASE.parent / "Final_Caves_of_QUD.owl"
GEN = BASE / "generated"
NS = "http://www.semanticweb.org/chris/ontologies/caverne-di-qud#"

def short(u): return u.split("#",1)[1] if "#" in u else u
def words(n): return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", n).replace("_"," ").lower().strip()

g = Graph(); g.parse(OWL_PATH)
iface = json.loads((GEN/"ontology_interface.json").read_text(encoding="utf-8"))

gliner_labels = []
for c in iface["classes"]:
    cu = URIRef(NS+c)
    com = g.value(cu, RDFS.comment)
    gliner_labels.append({"label": words(c), "class_short": c,
                          "description": (str(com)[:240] if com else "")})

glirel_labels = []
for r in iface["relation_vocabulary"]:
    if r["kind"] != "object_property": continue
    glirel_labels.append({"label": words(r["short"]), "predicate_short": r["short"],
                          "uri": r["uri"], "domain": r["domain"], "range": r["range"],
                          "inverse": r["inverse"]})

(GEN/"gliner_labels.json").write_text(json.dumps(gliner_labels, indent=2, ensure_ascii=False), encoding="utf-8")
(GEN/"glirel_labels.json").write_text(json.dumps(glirel_labels, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"GLiNER entity-type labels: {len(gliner_labels)}")
print(f"GLiREL relation labels:    {len(glirel_labels)}")
print("sample gliner:", json.dumps(gliner_labels[0], ensure_ascii=False)[:120])
print("sample glirel:", json.dumps(glirel_labels[0], ensure_ascii=False)[:160])
