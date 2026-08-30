"""Emit a compact ontology 'schema card' (hierarchy + constraints + individuals) for LLM prompting."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from rdflib import Graph, URIRef, RDF, RDFS, OWL
from rdflib.collection import Collection

BASE = Path(__file__).resolve().parents[1]
OWL_PATH = BASE.parent / "Final_Caves_of_QUD.owl"
OUT = BASE / "generated" / "ontology_schema_card.md"
NS = "http://www.semanticweb.org/chris/ontologies/caverne-di-qud#"

def s(u): return u.split("#",1)[1] if isinstance(u,(URIRef,str)) and "#" in u else str(u)

g = Graph(); g.parse(OWL_PATH)

def resolve_classes(node):
    if isinstance(node, URIRef): return [s(node)]
    out=[]
    u=g.value(node, OWL.unionOf)
    if u is not None:
        out=[s(m) for m in Collection(g,u) if isinstance(m,URIRef)]
    return out

classes=sorted(s(c) for c in g.subjects(RDF.type, OWL.Class) if isinstance(c,URIRef))
obj=sorted(s(p) for p in g.subjects(RDF.type, OWL.ObjectProperty))
dat=sorted(s(p) for p in g.subjects(RDF.type, OWL.DatatypeProperty))
inds=sorted(s(i) for i in g.subjects(RDF.type, OWL.NamedIndividual))

superclasses=defaultdict(list); disjoint=set(); equiv=[]
for c in classes:
    cu=URIRef(NS+c)
    for sup in g.objects(cu, RDFS.subClassOf):
        if isinstance(sup, URIRef): superclasses[c].append(s(sup))
    for d in g.objects(cu, OWL.disjointWith):
        if isinstance(d, URIRef): disjoint.add(tuple(sorted((c,s(d)))))
    for eq in g.objects(cu, OWL.equivalentClass):
        members=resolve_classes(eq)
        if members: equiv.append((c, members))

def prop_dr(p):
    cu=URIRef(NS+p)
    dom=[]; rng=[]
    for d in g.objects(cu, RDFS.domain): dom+=resolve_classes(d)
    for r in g.objects(cu, RDFS.range): rng+=resolve_classes(r)
    inv=g.value(cu, OWL.inverseOf)
    return sorted(set(dom)), sorted(set(rng)), (s(inv) if isinstance(inv,URIRef) else None)

def dt_range(p):
    cu=URIRef(NS+p); out=[]
    for r in g.objects(cu, RDFS.range):
        if isinstance(r, URIRef): out.append(s(r).split("#")[-1] if "#" in str(r) else str(r).split("/")[-1])
        else:
            one=g.value(r, OWL.oneOf)
            if one is not None: out.append("{"+", ".join(str(x) for x in Collection(g,one))+"}")
    dom=[]
    for d in g.objects(cu, RDFS.domain): dom+=resolve_classes(d)
    return sorted(set(dom)), (" | ".join(out) if out else "literal")

ind_types=defaultdict(list)
for i in inds:
    for t in g.objects(URIRef(NS+i), RDF.type):
        if isinstance(t,URIRef) and s(t) in classes: ind_types[s(t)].append(i)

L=[]
L.append("# Caves of Qud — Ontology Schema Card\n")
L.append(f"Namespace: `{NS}`  ·  {len(classes)} classes · {len(obj)} object properties · {len(dat)} datatype properties · {len(inds)} individuals\n")

L.append("## Class hierarchy (subClassOf)")
roots=[c for c in classes if not superclasses[c]]
def tree(c, depth):
    L.append("  "*depth + f"- {c}")
    for ch in sorted(k for k,v in superclasses.items() if c in v): tree(ch, depth+1)
for r in roots: tree(r,0)
L.append("")
if equiv:
    L.append("## Complete (equivalentClass) — for a reasoner, not SHACL")
    for c,m in equiv: L.append(f"- {c} ≡ {' ⊔ '.join(m)}")
    L.append("")
if disjoint:
    L.append("## Disjoint classes")
    for a,b in sorted(disjoint): L.append(f"- {a} ⊥ {b}")
    L.append("")

L.append("## Object properties  (domain → range)")
for p in obj:
    d,r,inv=prop_dr(p)
    L.append(f"- **{p}**: {' ⊔ '.join(d) or '?'} → {' ⊔ '.join(r) or '?'}" + (f"  (inverse: {inv})" if inv else ""))
L.append("")
L.append("## Datatype properties  (domain → type)")
for p in dat:
    d,rng=dt_range(p)
    L.append(f"- **{p}**: {' ⊔ '.join(d) or '?'} → {rng}")
L.append("")
L.append("## Individuals by class")
for c in sorted(ind_types):
    names=sorted(set(ind_types[c]))
    L.append(f"- **{c}** ({len(names)}): " + ", ".join(n.replace('_',' ') for n in names))
L.append("")
L.append("## Constraints for generation")
L.append("- Every triple MUST satisfy the property's domain and range above.")
L.append("- Respect disjoint classes (an individual cannot be both, e.g. Fighter ⊥ Trader).")
L.append("- Use ONLY the individuals listed; relations only between type-compatible individuals.")
L.append("- Inverse pairs are equivalent facts (state either direction).")

OUT.write_text("\n".join(L), encoding="utf-8")
print("wrote", OUT)
print(f"{len(classes)} classes, {len(obj)} obj props, {len(dat)} dt props, {len(inds)} individuals, {len(disjoint)} disjoint, {len(equiv)} complete")
