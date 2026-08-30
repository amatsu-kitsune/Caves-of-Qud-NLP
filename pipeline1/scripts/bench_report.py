"""
Pipeline 1 - FULL BENCHMARK REPORT.

Assembles the six deliverable table families into one markdown document:

  1. per-story TP/FP/FN + P/R/F1        (one table per model)
  2. model comparison summary            (one row per model)
  3. verifier confusion matrices         (train / val / test, per model)
  4. dataset split table                 (POS/NEG + per-relation counts)
  5. verifier summary per split          (accuracy / precision / recall / F1)
  6. ontology structure tables           (classes, properties, individuals)

Families 1-2 are re-read from a previous `bench_run.py` pass; 4 and 6 are read
straight from the generated artefacts. Only 3 and 5 need model inference, so
their predictions are cached in `verifier_splits_eval.json` and the whole report
can be rebuilt without a GPU via --no-infer.

Examples (in the env that has the models):
  conda activate MEHMET && cd pipeline1/scripts
  python bench_report.py                      # infer on the 3 splits, write everything
  python bench_report.py --no-infer           # rebuild the doc from caches only
  python bench_report.py --models t5_base,distilbert_base_uncased
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import bench_lib as bl

SPLITS = ("train", "val", "test")


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(":---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def disp(name):
    return "spaCy (baseline)" if name == "spacy" else name


def model_order(present):
    """spaCy baseline first, then the transformer zoo, then anything else."""
    order = ["spacy"] + [bl.model_dir_for(hf).name.replace("verifier_", "") for _, hf in bl.BERT_ZOO]
    ranked = [n for n in order if n in present]
    return ranked + [n for n in present if n not in ranked]


def eval_split(adapter, rows):
    """Confusion matrix + metrics for one split, VALID treated as the positive class."""
    preds = adapter.label_texts([r["text"] for r in rows])
    gold = [r["label"] for r in rows]

    cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for g, p in zip(gold, preds):
        if g == "INVALID" and p == "INVALID":
            cm["tn"] += 1
        elif g == "INVALID" and p == "VALID":
            cm["fp"] += 1
        elif g == "VALID" and p == "INVALID":
            cm["fn"] += 1
        else:
            cm["tp"] += 1

    acc = (cm["tp"] + cm["tn"]) / max(1, len(rows))
    p, r, f = prf_from_counts(cm["tp"], cm["fp"], cm["fn"])

    macro = []
    for lab in ("VALID", "INVALID"):
        tp = sum(g == lab and q == lab for g, q in zip(gold, preds))
        fp = sum(g != lab and q == lab for g, q in zip(gold, preds))
        fn = sum(g == lab and q != lab for g, q in zip(gold, preds))
        macro.append(prf_from_counts(tp, fp, fn))

    return {
        "n_rows": len(rows),
        "confusion": cm,
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "macro_precision": round(sum(m[0] for m in macro) / 2, 4),
        "macro_recall": round(sum(m[1] for m in macro) / 2, 4),
        "macro_f1": round(sum(m[2] for m in macro) / 2, 4),
    }


def run_split_eval(specs, split_rows):
    out = {}
    for spec in specs:
        name = spec["name"]
        print(f"\n=== {name} ({spec['type']}) ===")
        try:
            adapter = bl.load_adapter(spec)
        except Exception as e:
            print("  ! could not load:", repr(e))
            continue
        out[name] = {"type": spec["type"]}
        for sp in SPLITS:
            rows = split_rows.get(sp) or []
            if not rows:
                continue
            res = eval_split(adapter, rows)
            out[name][sp] = res
            cm = res["confusion"]
            print(f"  {sp:5s} n={res['n_rows']:5d} acc={res['accuracy']:.4f} "
                  f"F1={res['f1']:.4f}  [tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}]")
        del adapter
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def section_per_story(per_txt_path, top_n):
    """1. per-story TP/FP/FN, one table per model."""
    if not per_txt_path.exists():
        return [f"_missing `{per_txt_path.name}` - run `bench_run.py` first._", ""]

    by_model = defaultdict(list)
    with per_txt_path.open(encoding="utf-8") as h:
        for row in csv.DictReader(h):
            by_model[row["model"]].append(row)

    md = []
    for name in model_order(list(by_model)):
        rows = sorted(by_model[name], key=lambda r: r["id"])
        shown = rows if top_n <= 0 else rows[:top_n]
        md += [f"### {disp(name)}", ""]
        table = [[r["id"], r["tier"], r["tp"], r["fp"], r["fn"],
                  pct(float(r["precision"])), pct(float(r["recall"])), pct(float(r["f1"]))]
                 for r in shown]

        tp = sum(int(r["tp"]) for r in rows)
        fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        p, r_, f = prf_from_counts(tp, fp, fn)
        table.append(["**TOTAL (micro)**", f"{len(rows)} txt", f"**{tp}**", f"**{fp}**", f"**{fn}**",
                      f"**{pct(p)}**", f"**{pct(r_)}**", f"**{pct(f)}**"])

        md += [md_table(["Story", "Tier", "TP", "FP", "FN", "Precision", "Recall", "F1"], table)]
        if top_n > 0 and len(rows) > top_n:
            md += ["", f"_showing first {top_n} of {len(rows)} texts; full data in "
                       f"`{per_txt_path.name}`. TOTAL row covers all {len(rows)}._"]
        md += [""]
    return md


def section_model_summary(results):
    """2. one row per model: e2e triple extraction + KG."""
    if not results:
        return ["_missing `benchmark_results.json` - run `bench_run.py` first._", ""]

    models = results.get("models", {})
    md = ["Micro-averaged over the whole evaluation corpus "
          f"(`{results.get('corpus')}`, {results.get('n_units')} texts).", ""]

    rows = []
    for name in model_order(list(models)):
        e2e = models[name].get("e2e") or {}
        allt = (e2e.get("per_tier") or {}).get("ALL") or {}
        kg = e2e.get("kg") or {}
        if not allt and not kg:
            continue
        rows.append([disp(name),
                     pct(allt.get("precision")), pct(allt.get("recall")), pct(allt.get("f1")),
                     pct(kg.get("kg_precision")), pct(kg.get("kg_recall")), pct(kg.get("kg_f1"))])
    md += [md_table(["Model", "Precision", "Recall", "F1-score",
                     "KG Precision", "KG Recall", "KG F1"], rows), ""]

    md += ["Per-tier breakdown of the end-to-end F1:", ""]
    tiers = []
    for m in models.values():
        for t in (m.get("e2e") or {}).get("per_tier", {}):
            if t not in tiers and t != "ALL":
                tiers.append(t)
    rows = []
    for name in model_order(list(models)):
        pt = (models[name].get("e2e") or {}).get("per_tier") or {}
        if not pt:
            continue
        rows.append([disp(name)] + [pct((pt.get(t) or {}).get("f1")) for t in tiers]
                    + [pct((pt.get("ALL") or {}).get("f1"))])
    md += [md_table(["Model"] + tiers + ["ALL"], rows), ""]
    return md


def section_confusion(split_eval):
    """3. confusion matrix per model per split."""
    if not split_eval:
        return ["_no split evaluation cached - run without `--no-infer`._", ""]
    md = []
    for name in model_order([k for k in split_eval if k != "_meta"]):
        entry = split_eval[name]
        md += [f"### {disp(name)}", ""]
        for sp in SPLITS:
            res = entry.get(sp)
            if not res:
                continue
            cm = res["confusion"]
            md += [f"**{sp.capitalize()}** — {res['n_rows']} rows", "",
                   md_table(["", "Pred. INVALID", "Pred. VALID"],
                            [["**Actual INVALID**", cm["tn"], cm["fp"]],
                             ["**Actual VALID**", cm["fn"], cm["tp"]]]),
                   ""]
    return md


def section_dataset_split(split_rows, top_rel):
    """4. dataset split distribution + per-relation counts."""
    if not any(split_rows.values()):
        return ["_missing `verifier_{train,val,test}.jsonl`._", ""]

    freq = Counter()
    for rows in split_rows.values():
        for r in rows:
            freq[r["candidate_relation"]] += 1
    rels = [r for r, _ in freq.most_common(top_rel)]

    table = []
    for sp in SPLITS:
        rows = split_rows.get(sp) or []
        if not rows:
            continue
        neg = sum(r["label"] == "INVALID" for r in rows)
        pos = sum(r["label"] == "VALID" for r in rows)
        per_rel = Counter(r["candidate_relation"] for r in rows)
        table.append([sp.capitalize(), len(rows), neg, pos] + [per_rel.get(x, 0) for x in rels])
    md = [md_table(["Split", "Rows", "NEG (INVALID)", "POS (VALID)"] + rels, table),
          "", f"_columns after POS are the {len(rels)} most frequent candidate relations "
              f"of {len(freq)} distinct._", ""]

    md += ["Negative examples by construction type:", ""]
    ntypes = sorted({r.get("neg_type") for rows in split_rows.values()
                     for r in rows if r["label"] == "INVALID" and r.get("neg_type")})
    table = []
    for sp in SPLITS:
        rows = split_rows.get(sp) or []
        if not rows:
            continue
        c = Counter(r.get("neg_type") for r in rows if r["label"] == "INVALID")
        k = Counter(r.get("rel_kind") for r in rows)
        table.append([sp.capitalize()] + [c.get(n, 0) for n in ntypes]
                     + [k.get("object", 0), k.get("datatype", 0)])
    md += [md_table(["Split"] + [f"neg: {n}" for n in ntypes] + ["object", "datatype"], table), ""]
    return md


def section_verifier_summary(split_eval):
    """5. accuracy / precision / recall / F1 per split per model."""
    if not split_eval:
        return ["_no split evaluation cached - run without `--no-infer`._", ""]
    md = ["Positive class = `VALID`; macro-F1 averages both classes.", ""]
    for name in model_order([k for k in split_eval if k != "_meta"]):
        entry = split_eval[name]
        rows = []
        for sp in SPLITS:
            res = entry.get(sp)
            if not res:
                continue
            rows.append([sp.capitalize(), res["n_rows"], pct(res["accuracy"]), pct(res["recall"]),
                         pct(res["precision"]), pct(res["f1"]), pct(res["macro_f1"])])
        if rows:
            md += [f"### {disp(name)}", "",
                   md_table(["Split", "Rows", "Accuracy", "Recall", "Precision", "F1", "Macro-F1"], rows), ""]
    return md


def section_ontology(iface):
    """6. structural tables of the ontology."""
    if not iface:
        return ["_missing `ontology_interface.json`._", ""]

    anc = iface.get("class_ancestors", {})
    children = defaultdict(list)
    for cls, parents in anc.items():
        for p in parents:
            if p != cls:
                children[p].append(cls)
    roots = sorted(c for c in iface.get("classes", []) if len(anc.get(c, [c])) <= 1)
    md = ["**Class hierarchy**", "",
          md_table(["Core class", "Subclasses"],
                   [[c, ", ".join(sorted(children.get(c, []))) or "-"] for c in roots]), ""]

    rv = iface.get("relation_vocabulary", [])
    obj = [r for r in rv if r["kind"] == "object_property"]
    dat = [r for r in rv if r["kind"] != "object_property"]
    md += ["**Object properties**", "",
           md_table(["Property", "Domain", "Range", "Inverse"],
                    [[r["short"], " ⊔ ".join(r["domain"]) or "-",
                      " ⊔ ".join(r["range"]) or "-", r.get("inverse") or "-"]
                     for r in sorted(obj, key=lambda x: x["short"])]), ""]
    if dat:
        md += ["**Datatype properties**", "",
               md_table(["Property", "Domain", "Range"],
                        [[r["short"], " ⊔ ".join(r["domain"]) or "-", " ⊔ ".join(r["range"]) or "-"]
                         for r in sorted(dat, key=lambda x: x["short"])]), ""]

    by_class = defaultdict(list)
    for ind, types in (iface.get("individual_types") or {}).items():
        for t in types:
            by_class[t].append(ind)
    if by_class:
        md += ["**Individuals per class**", "",
               md_table(["Class", "N", "Individuals"],
                        [[c, len(v), ", ".join(sorted(x.replace("_", " ") for x in v))]
                         for c, v in sorted(by_class.items())]), ""]
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="zoo",
                    help="'zoo' | 'auto' | comma list of model dir basenames")
    ap.add_argument("--corpus", default="tiered_test",
                    help="corpus tag of the bench_run pass to read per-story metrics from")
    ap.add_argument("--no-infer", action="store_true",
                    help="skip model inference, reuse cached verifier_splits_eval.json")
    ap.add_argument("--top-stories", type=int, default=25,
                    help="rows per per-story table (0 = all)")
    ap.add_argument("--top-relations", type=int, default=8,
                    help="relation columns in the dataset-split table")
    ap.add_argument("--out", type=Path, default=bl.OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    split_rows = {sp: (bl._jsonl(bl.GEN / f"verifier_{sp}.jsonl")
                       if (bl.GEN / f"verifier_{sp}.jsonl").exists() else [])
                  for sp in SPLITS}

    cache_path = args.out / "verifier_splits_eval.json"
    if args.no_infer:
        split_eval = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        if not split_eval:
            print(f"! no cache at {cache_path}; sections 3 and 5 will be empty")
    else:
        import bench_run
        specs = bench_run.resolve_specs(args.models)
        specs = [s for s in specs if s["name"] != "passthrough"]
        print("verifiers:", [s["name"] for s in specs])
        split_eval = run_split_eval(specs, split_rows)
        split_eval["_meta"] = {"rows": {sp: len(split_rows[sp]) for sp in SPLITS}}
        cache_path.write_text(json.dumps(split_eval, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote", cache_path)

    res_path = args.out / "benchmark_results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    iface_path = bl.GEN / "ontology_interface.json"
    iface = json.loads(iface_path.read_text(encoding="utf-8")) if iface_path.exists() else {}
    per_txt = args.out / f"benchmark_per_txt_{args.corpus}.csv"

    md = ["# Pipeline 1 - Full Benchmark Report", "",
          f"- **Chain:** txt -> GLiNER -> GLiREL -> verifier -> SHACL -> KG",
          f"- **Stages 1&2:** {results.get('stage12', 'n/a')}",
          f"- **End-to-end corpus:** `{results.get('corpus', args.corpus)}` "
          f"({results.get('n_units', '?')} texts)",
          f"- **Verifier rows:** " + " · ".join(f"{sp} {len(split_rows[sp])}" for sp in SPLITS),
          "",
          "## 1. Per-story performance (TP / FP / FN)", ""]
    md += section_per_story(per_txt, args.top_stories)
    md += ["## 2. Model comparison summary", ""] + section_model_summary(results)
    md += ["## 3. Verifier confusion matrices", ""] + section_confusion(split_eval)
    md += ["## 4. Dataset split", ""] + section_dataset_split(split_rows, args.top_relations)
    md += ["## 5. Verifier summary per split", ""] + section_verifier_summary(split_eval)
    md += ["## 6. Ontology structure", ""] + section_ontology(iface)

    path = args.out / "benchmark_full_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
