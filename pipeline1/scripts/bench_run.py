"""
Pipeline 1 - BENCHMARK runner.

Benchmarks stage-3 verifiers (different BERT families + the spaCy baseline +
a no-verifier reference) on TWO levels, holding stages 1/2/4/5 fixed:

  INTRINSIC : VALID/INVALID classification on generated/verifier_test.jsonl
              (accuracy, macro-F1, and rejection recall per hard-negative type)

  END-TO-END: full chain  txt -> GLiNER -> GLiREL -> <verifier> -> SHACL -> KG
              over the tiered corpus, reporting per stage:
                entity_recall (stage 1) | candidate_recall ceiling (stage 2)
                triple Precision/Recall/F1 (stage 3) | KG-level P/R/F1 (stage 5)
              broken down per complexity tier.

Outputs (pipeline1/outputs/):
  benchmark_results.json     full machine-readable results
  benchmark_report.md        human-readable comparison tables
  kg_<model>.ttl             assembled knowledge graph per verifier

Examples (in the env that has the models / GLiNER / GLiREL):
  conda activate MEHMET && cd pipeline1/scripts
  python bench_run.py                                   # auto-discover models, fallback taggers
  python bench_run.py --use-gliner --use-glirel         # real stage 1 & 2
  python bench_run.py --models bert_base_uncased,roberta_base,spacy --corpus tiered
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import infer_pipeline as ip
import bench_lib as bl


def load_taggers(use_gliner, use_glirel):
    gliner_model = glirel_model = None
    if use_gliner:
        from gliner import GLiNER
        gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    if use_glirel:
        from glirel import GLiREL
        _orig = GLiREL._from_pretrained.__func__

        def _compat(cls, *a, **kw):
            kw.setdefault("proxies", None)
            kw.setdefault("resume_download", False)
            return _orig(cls, *a, **kw)
        GLiREL._from_pretrained = classmethod(_compat)
        glirel_model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    return gliner_model, glirel_model


def resolve_specs(arg):
    found = {s["name"]: s for s in bl.discover_models()}
    specs = []
    if arg == "auto":
        specs = list(found.values())
    elif arg == "zoo":
        for _, hf in bl.BERT_ZOO:
            nm = bl.model_dir_for(hf).name.replace("verifier_", "")
            if nm in found:
                specs.append(found[nm])
            else:
                print(f"  ! zoo model '{nm}' not trained yet (skipped)")
        if "spacy" in found:
            specs.append(found["spacy"])
    else:
        for name in [x.strip() for x in arg.split(",") if x.strip()]:
            if name in ("passthrough", "none"):
                continue
            if name in found:
                specs.append(found[name])
            else:
                print(f"  ! model '{name}' not found in models/ (skipped)")
    specs.append({"name": "passthrough", "type": "none", "path": None})
    return specs


def intrinsic_eval(adapter, rows):
    preds = adapter.label_texts([r["text"] for r in rows])
    gold = [r["label"] for r in rows]
    acc = sum(g == p for g, p in zip(gold, preds)) / max(1, len(rows))
    precs, recs, f1s = {}, {}, {}
    for lab in ("VALID", "INVALID"):
        tp = sum(g == lab and p == lab for g, p in zip(gold, preds))
        fp = sum(g != lab and p == lab for g, p in zip(gold, preds))
        fn = sum(g == lab and p != lab for g, p in zip(gold, preds))
        precs[lab] = tp / (tp + fp) if (tp + fp) else 0.0
        recs[lab] = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[lab] = (2 * precs[lab] * recs[lab] / (precs[lab] + recs[lab])
                    if (precs[lab] + recs[lab]) else 0.0)
    neg_tot, neg_hit = Counter(), Counter()
    valid_tot = valid_hit = 0
    for r, p in zip(rows, preds):
        if r["label"] == "INVALID":
            nt = r.get("neg_type") or "other"
            neg_tot[nt] += 1
            neg_hit[nt] += (p == "INVALID")
        else:
            valid_tot += 1
            valid_hit += (p == "VALID")
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(sum(precs.values()) / 2, 4),
        "macro_recall": round(sum(recs.values()) / 2, 4),
        "macro_f1": round(sum(f1s.values()) / 2, 4),
        "valid_recall": round(valid_hit / valid_tot, 4) if valid_tot else None,
        "reject_recall": {nt: round(neg_hit[nt] / neg_tot[nt], 4) for nt in neg_tot},
        "n_rows": len(rows),
    }


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="tiered_test",
                    choices=["tiered_train", "tiered_val", "tiered_test", "tiered"])
    ap.add_argument("--models", default="zoo",
                    help="'zoo' (BERT_ZOO + spaCy) | 'auto' (all on disk) | comma list of dir basenames")
    ap.add_argument("--eval", choices=["intrinsic", "e2e", "both"], default="both")
    ap.add_argument("--use-gliner", action="store_true")
    ap.add_argument("--use-glirel", action="store_true")
    ap.add_argument("--stage2-union", action=argparse.BooleanOptionalAction, default=True,
                    help="union GLiREL candidates with domain/range pairwise generation "
                         "(--no-stage2-union = pure GLiREL)")
    ap.add_argument("--glirel-topk", type=int, default=3,
                    help="GLiREL top-k relation labels per pair (was 1)")
    ap.add_argument("--shapes", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap eval units (debug)")
    ap.add_argument("--out", type=Path, default=bl.OUT)
    args = ap.parse_args()

    onto = ip.Ontology()
    specs = resolve_specs(args.models)
    print("verifiers:", [s["name"] for s in specs])

    gliner_model, glirel_model = load_taggers(args.use_gliner, args.use_glirel)
    if args.use_glirel:
        stage2 = ("real+pairwise-union" if args.stage2_union else "real") + f" topk={args.glirel_topk}"
    else:
        stage2 = "pairwise-fallback"
    stage12 = (f"GLiNER={'real+dict-union' if args.use_gliner else 'dict-fallback'} | "
               f"GLiREL={stage2}")

    test_rows = bl._jsonl(bl.GEN / "verifier_test.jsonl") if (bl.GEN / "verifier_test.jsonl").exists() else []
    units = bl.load_units(args.corpus)
    if args.limit:
        units = units[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    results = {"stage12": stage12, "corpus": args.corpus,
               "n_test_rows": len(test_rows), "n_units": len(units), "models": {}}
    per_txt_rows = []

    for spec in specs:
        name = spec["name"]
        print(f"\n=== {name} ({spec['type']}) ===")
        try:
            adapter = bl.load_adapter(spec)
        except Exception as e:
            print("  ! could not load:", repr(e))
            continue
        entry = {"type": spec["type"]}

        if args.eval in ("intrinsic", "both") and test_rows:
            entry["intrinsic"] = intrinsic_eval(adapter, test_rows)
            print("  intrinsic:", json.dumps(entry["intrinsic"]))

        if args.eval in ("e2e", "both"):
            sc = bl.score_units(units, onto, adapter,
                                use_gliner=args.use_gliner, gliner_model=gliner_model,
                                use_glirel=args.use_glirel, glirel_model=glirel_model,
                                collect_per_unit=True,
                                stage2_union=args.stage2_union, glirel_topk=args.glirel_topk)
            for row in sc.get("per_unit") or []:
                per_txt_rows.append({"model": name, **row})
            kg_pred, kg_gold = sc["kg_pred"], sc["kg_gold"]
            kg_pre = bl.kg_metrics(kg_pred, kg_gold)
            if args.shapes and Path(args.shapes).exists():
                kept, removed, conforms = bl.shacl_filter_triples(kg_pred, onto, args.shapes)
            else:
                kept, removed, conforms = kg_pred, [], None
            kg_post = bl.kg_metrics(kept, kg_gold)
            diff = bl.write_eval_graphs(kept, kg_gold, onto, args.out, name)
            entry["e2e"] = {"per_tier": sc["per_tier"], "seconds": sc["seconds"],
                            "per_txt": sc.get("per_unit"),
                            "kg_pre_shacl": kg_pre,
                            "kg": {**kg_post, "shacl_removed": len(removed),
                                   "shacl_conforms": conforms, **diff}}
            print("  e2e stage3 ALL:", json.dumps(sc["per_tier"].get("ALL", {})))
            print(f"  shacl: removed={len(removed)} conforms={conforms}")
            print("  kg(post-shacl):", json.dumps(kg_post))

        results["models"][name] = entry
        del adapter
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    (args.out / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results, args.out / "benchmark_report.md")
    write_prf_report(results, args.out / "benchmark_prf.md")
    print("\nwrote", args.out / "benchmark_results.json")
    print("wrote", args.out / "benchmark_report.md")
    print("wrote", args.out / "benchmark_prf.md")

    if per_txt_rows:
        per_txt_path = args.out / f"benchmark_per_txt_{args.corpus}.csv"
        fields = ["model", "id", "tier", "n_gold", "n_pred", "tp", "fp", "fn",
                  "precision", "recall", "f1", "exact_match", "entity_recall", "n_candidates"]
        with per_txt_path.open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_txt_rows)
        print("wrote", per_txt_path)


def write_prf_report(results, path):
    """Slim deliverable: spaCy baseline vs the transformer zoo, P/R/F1 ONLY.
    Two views: the stage-3 verifier as a classifier (macro P/R/F1 on verifier_test)
    and the full pipeline as a triple extractor (micro P/R/F1 on the test corpus)."""
    order = ["spacy"] + [bl.model_dir_for(hf).name.replace("verifier_", "") for _, hf in bl.BERT_ZOO]
    present = [n for n in order if n in results["models"]]

    def disp(n):
        return "spaCy (baseline)" if n == "spacy" else n

    md = ["# Benchmark - spaCy baseline vs Transformers (P / R / F1)",
          "",
          f"- corpus (end-to-end): `{results['corpus']}`  |  stages 1&2: {results['stage12']}",
          f"- verifier test rows: {results['n_test_rows']}  |  e2e units: {results['n_units']}",
          ""]

    if any("intrinsic" in results["models"][n] for n in present):
        md += ["## A. Verifier classification - VALID/INVALID (macro, on verifier_test)", ""]
        rows = []
        for n in present:
            it = results["models"][n].get("intrinsic")
            if it:
                rows.append([disp(n), it["macro_precision"], it["macro_recall"], it["macro_f1"]])
        md += [md_table(["model", "Precision", "Recall", "F1"], rows), ""]

    if any("e2e" in results["models"][n] for n in present):
        md += ["## B. Knowledge graph - post-SHACL, deduplicated (predicted vs expected)", ""]
        rows = []
        for n in present:
            kg = results["models"][n].get("e2e", {}).get("kg")
            if kg:
                rows.append([disp(n), kg["kg_precision"], kg["kg_recall"], kg["kg_f1"]])
        md += [md_table(["model", "Precision", "Recall", "F1"], rows),
               "",
               "_per-model graphs in `kg_<model>_predicted.ttl` vs `kg_expected.ttl`; "
               "TP/FP/FN in `kg_<model>_diff.tsv` and `kg_<model>_labeled.ttl`._", ""]

    path.write_text("\n".join(md), encoding="utf-8")


def write_report(results, path):
    md = ["# Pipeline 1 - Verifier Benchmark",
          "",
          f"- **Stage 1&2:** {results['stage12']}",
          f"- **Corpus:** {results['corpus']}  ({results['n_units']} units, "
          f"{results['n_test_rows']} intrinsic test rows)",
          "- **Chain:** txt -> GLiNER -> GLiREL -> verifier -> SHACL -> KG",
          ""]

    has_intr = any("intrinsic" in m for m in results["models"].values())
    if has_intr:
        md += ["## Stage 3 - intrinsic VALID/INVALID (verifier_test)", ""]
        neg_types = sorted({nt for m in results["models"].values()
                            for nt in m.get("intrinsic", {}).get("reject_recall", {})})
        headers = ["model", "acc", "macroF1", "validRec"] + [f"rej:{nt}" for nt in neg_types]
        rows = []
        for name, m in results["models"].items():
            it = m.get("intrinsic")
            if not it:
                continue
            rows.append([name, it["accuracy"], it["macro_f1"], it["valid_recall"]]
                        + [it["reject_recall"].get(nt, "-") for nt in neg_types])
        md += [md_table(headers, rows), ""]

    has_e2e = any("e2e" in m for m in results["models"].values())
    if has_e2e:
        tiers = []
        for m in results["models"].values():
            for t in m.get("e2e", {}).get("per_tier", {}):
                if t not in tiers:
                    tiers.append(t)
        tiers = [t for t in tiers if t != "ALL"] + (["ALL"] if any(
            "ALL" in m.get("e2e", {}).get("per_tier", {}) for m in results["models"].values()) else [])
        md += ["## End-to-end triple extraction (per tier)", ""]
        for tier in tiers:
            md += [f"### tier: `{tier}`", ""]
            headers = ["model", "entRec(s1)", "candCeil(s2)", "P", "R", "F1", "avgCands"]
            rows = []
            for name, m in results["models"].items():
                pt = m.get("e2e", {}).get("per_tier", {}).get(tier)
                if not pt:
                    continue
                rows.append([name, pt["entity_recall"], pt["candidate_recall_ceiling"],
                             pt["precision"], pt["recall"], pt["f1"], pt["avg_candidates_per_unit"]])
            md += [md_table(headers, rows), ""]

        md += ["## Stage 4-5 - SHACL filter + knowledge graph (post-SHACL) vs expected", ""]
        headers = ["model", "F1_preSHACL", "shaclRemoved", "conforms",
                   "kg_P", "kg_R", "kg_F1", "TP", "FP", "FN"]
        rows = []
        for name, m in results["models"].items():
            e = m.get("e2e")
            if not e:
                continue
            kg = e["kg"]
            pre = e.get("kg_pre_shacl", {})
            rows.append([name, pre.get("kg_f1"), kg.get("shacl_removed"), kg.get("shacl_conforms"),
                         kg["kg_precision"], kg["kg_recall"], kg["kg_f1"],
                         kg["tp"], kg["fp"], kg["fn"]])
        md += [md_table(headers, rows), ""]

    md += ["---",
           "_entRec = stage-1 entity recall; candCeil = stage-2 candidate recall "
           "(max achievable R for any verifier); P/R/F1 = post-verifier triples; "
           "`passthrough` = no verifier (accept all candidates)._",
           "",
           f"_Per-txt breakdown (one row per model x input text): "
           f"`benchmark_per_txt_{results['corpus']}.csv`._"]
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
