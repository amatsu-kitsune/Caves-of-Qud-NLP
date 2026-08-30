"""
Pipeline 1 - Fase 5: train the supervised VERIFIER (BERT-like, binary VALID/INVALID).
LOCAL script (no Colab/Drive). Run inside your conda env that has torch + transformers.

    conda activate MEHMET
    cd pipeline1/scripts
    python train_verifier.py                      # roberta-base (default)
    python train_verifier.py --model bert-base-uncased --epochs 4

Reads : generated/verifier_train.jsonl, generated/verifier_val.jsonl
Writes: models/verifier_<encoder>/best_model/   (+ metrics.json)

Input format (one candidate per row):
  {"text": "[REL] giveQuest [/REL] [E1]Otho[/E1] gives ... [E2]A Call to Arms[/E2].", "label": "VALID"}
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
OUT = BASE / "outputs"
SPECIAL_TOKENS = ["[REL]", "[/REL]", "[E1]", "[/E1]", "[E2]", "[/E2]"]


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


class JsonlDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def encode(rows, tok, label2id, max_len):
    enc = tok([r["text"] for r in rows], truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    y = torch.tensor([label2id[r["label"]] for r in rows], dtype=torch.long)
    return enc, y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        labels = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds += logits.argmax(-1).cpu().tolist()
        golds += labels.tolist()
    return np.array(golds), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="LR warmup as a fraction of total steps (e.g. 0.1). "
                         "RoBERTa/XLNet/BigBird/T5 need this or they collapse to majority class.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| encoder:", args.model)

    train_rows = load_jsonl(GEN / "verifier_train.jsonl")
    val_rows = load_jsonl(GEN / "verifier_val.jsonl")
    labels = sorted({r["label"] for r in train_rows + val_rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print("labels:", label2id, "| train:", len(train_rows), "| val:", len(val_rows))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    train_enc, train_y = encode(train_rows, tok, label2id, args.max_len)
    val_enc, val_y = encode(val_rows, tok, label2id, args.max_len)
    train_loader = DataLoader(JsonlDataset(train_enc, train_y), batch_size=args.batch,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(JsonlDataset(val_enc, val_y), batch_size=64,
                            shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels))
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.resize_token_embeddings(len(tok))
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup * total_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"total_steps={total_steps} | warmup_steps={warmup_steps} | lr={args.lr}")

    out_dir = BASE / "models" / f"verifier_{args.model.split('/')[-1].replace('-', '_')}"
    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    epoch_log = []

    label_ids = list(range(len(labels)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += out.loss.item()
        g, p = evaluate(model, val_loader, device)
        acc = float((g == p).mean())
        train_loss = running / len(train_loader)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            g, p, labels=label_ids, average="macro", zero_division=0)
        per_p, per_r, per_f, per_s = precision_recall_fscore_support(
            g, p, labels=label_ids, average=None, zero_division=0)
        rec = {"encoder": args.model, "epoch": epoch,
               "train_loss": round(float(train_loss), 6),
               "val_accuracy": round(acc, 6),
               "val_macro_precision": round(float(macro_p), 6),
               "val_macro_recall": round(float(macro_r), 6),
               "val_macro_f1": round(float(macro_f1), 6)}
        for i in label_ids:
            lab = id2label[i]
            rec[f"val_{lab}_precision"] = round(float(per_p[i]), 6)
            rec[f"val_{lab}_recall"] = round(float(per_r[i]), 6)
            rec[f"val_{lab}_f1"] = round(float(per_f[i]), 6)
            rec[f"val_{lab}_support"] = int(per_s[i])
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={acc:.4f}  val_macroP={macro_p:.4f}  val_macroR={macro_r:.4f}  "
              f"val_macroF1={macro_f1:.4f}")
        is_best = macro_f1 > best_f1
        rec["is_best"] = bool(is_best)
        epoch_log.append(rec)
        if is_best:
            best_f1 = macro_f1
            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
            print("  saved best ->", best_dir)

    best = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    g, p = evaluate(best, val_loader, device)
    target_names = [id2label[i] for i in range(len(labels))]
    report = classification_report(g, p, target_names=target_names, digits=4)
    cm = confusion_matrix(g, p).tolist()
    print("\n=== VAL classification report (best) ===")
    print(report)
    print("confusion_matrix [rows=true, cols=pred] order", target_names, ":", cm)

    (out_dir / "metrics.json").write_text(json.dumps({
        "encoder": args.model, "best_val_macro_f1": best_f1,
        "labels": target_names, "confusion_matrix": cm,
        "report": classification_report(g, p, target_names=target_names, digits=4, output_dict=True),
    }, indent=2), encoding="utf-8")
    print("\nSaved:", best_dir, "and", out_dir / "metrics.json")

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
    print("Saved per-epoch log:", out_dir / "training_log.csv",
          "| global:", glob)


if __name__ == "__main__":
    main()
