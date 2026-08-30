"""
Pipeline 1 - stage-3 BASELINE verifier: spaCy textcat (VALID / INVALID).

This is the non-transformer baseline the BERT families are benchmarked against.
Same task, same input string, same label scheme as train_verifier.py:

    {"text": "[REL] giveQuest [/REL] [E1]Otho[/E1] gives ... [E2]A Call to Arms[/E2].",
     "label": "VALID"}

Reads : generated/verifier_train.jsonl, verifier_val.jsonl, verifier_test.jsonl
Writes: models/verifier_spacy/            (loadable with spacy.load)
        models/verifier_spacy_metrics.json

Run (in the env that has spaCy):
    conda activate MEHMET
    cd pipeline1/scripts
    python bench_spacy_train.py --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
MODELS = BASE / "models"
OUT = BASE / "outputs"
LABELS = ["VALID", "INVALID"]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def to_examples(nlp, rows):
    from spacy.training import Example
    out = []
    for r in rows:
        cats = {l: (1.0 if r["label"] == l else 0.0) for l in LABELS}
        out.append(Example.from_dict(nlp.make_doc(r["text"]), {"cats": cats}))
    return out


def predict_labels(nlp, rows):
    preds = []
    for doc in nlp.pipe([r["text"] for r in rows]):
        preds.append(max(doc.cats, key=doc.cats.get) if doc.cats else "INVALID")
    return preds


def report(rows, preds):
    gold = [r["label"] for r in rows]
    acc = sum(g == p for g, p in zip(gold, preds)) / max(1, len(gold))
    per = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, preds))
        fp = sum(g != lab and p == lab for g, p in zip(gold, preds))
        fn = sum(g == lab and p != lab for g, p in zip(gold, preds))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": tp + fn}
    n = len(LABELS)
    return {"accuracy": round(acc, 4),
            "macro_precision": round(sum(per[l]["precision"] for l in LABELS) / n, 4),
            "macro_recall": round(sum(per[l]["recall"] for l in LABELS) / n, 4),
            "macro_f1": round(sum(per[l]["f1"] for l in LABELS) / n, 4),
            "per_label": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    import spacy
    from spacy.util import minibatch, compounding

    random.seed(args.seed)
    spacy.util.fix_random_seed(args.seed)

    train = load_jsonl(GEN / "verifier_train.jsonl")
    val = load_jsonl(GEN / "verifier_val.jsonl")
    test_path = GEN / "verifier_test.jsonl"
    test = load_jsonl(test_path) if test_path.exists() else val
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    for lab in LABELS:
        textcat.add_label(lab)

    train_examples = to_examples(nlp, train)
    optimizer = nlp.initialize(lambda: train_examples)

    best_f1, out_dir = -1.0, MODELS / "verifier_spacy"
    epoch_log = []
    ord_labels = sorted(LABELS)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=compounding(8.0, 64.0, 1.5)):
            nlp.update(batch, sgd=optimizer, drop=args.dropout, losses=losses)
        vmetrics = report(val, predict_labels(nlp, val))
        print(f"epoch {epoch:02d}/{args.epochs}  loss={losses.get('textcat', 0):.3f}  "
              f"val_acc={vmetrics['accuracy']:.4f}  val_macroF1={vmetrics['macro_f1']:.4f}")
        rec = {"encoder": "spacy-textcat", "epoch": epoch,
               "train_loss": round(float(losses.get("textcat", 0.0)), 6),
               "val_accuracy": vmetrics["accuracy"],
               "val_macro_precision": vmetrics["macro_precision"],
               "val_macro_recall": vmetrics["macro_recall"],
               "val_macro_f1": vmetrics["macro_f1"]}
        for lab in ord_labels:
            pl = vmetrics["per_label"][lab]
            rec[f"val_{lab}_precision"] = pl["precision"]
            rec[f"val_{lab}_recall"] = pl["recall"]
            rec[f"val_{lab}_f1"] = pl["f1"]
            rec[f"val_{lab}_support"] = pl["support"]
        is_best = vmetrics["macro_f1"] > best_f1
        rec["is_best"] = bool(is_best)
        epoch_log.append(rec)
        if is_best:
            best_f1 = vmetrics["macro_f1"]
            out_dir.mkdir(parents=True, exist_ok=True)
            nlp.to_disk(out_dir)

    nlp = spacy.load(out_dir)
    test_metrics = report(test, predict_labels(nlp, test))
    val_metrics = report(val, predict_labels(nlp, val))
    print("\n=== spaCy baseline ===")
    print("VAL :", val_metrics)
    print("TEST:", test_metrics)

    (MODELS / "verifier_spacy_metrics.json").write_text(json.dumps({
        "encoder": "spacy-textcat", "best_val_macro_f1": round(best_f1, 4),
        "val": val_metrics, "test": test_metrics,
    }, indent=2), encoding="utf-8")
    print("saved ->", out_dir)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(epoch_log[0].keys()) if epoch_log else []
    (out_dir / "training_log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in epoch_log), encoding="utf-8")
    with (out_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(epoch_log)
    glob = OUT / "training_log.csv"
    write_header = not glob.exists()
    with glob.open("a", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(epoch_log)
    print("Saved per-epoch log:", out_dir / "training_log.csv", "| global:", glob)


if __name__ == "__main__":
    main()
