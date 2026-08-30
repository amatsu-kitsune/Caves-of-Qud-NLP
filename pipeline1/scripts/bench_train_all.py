"""
Train the whole stage-3 verifier zoo for the benchmark:
  - every BERT family in bench_lib.BERT_ZOO  (via train_verifier.py)
  - the spaCy baseline                       (via bench_spacy_train.py)

Already-trained encoders are skipped unless --force. Run in the env that has
torch + transformers + spaCy (MEHMET).

    conda activate MEHMET
    cd pipeline1/scripts
    python bench_train_all.py                       # all families + spaCy, skip existing
    python bench_train_all.py --only roberta-base,distilbert-base-uncased
    python bench_train_all.py --epochs 4 --batch 16 --no-spacy --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import bench_lib as bl

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of HF ids to train (default: all)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.1,
                    help="LR warmup fraction. REQUIRED (>0) or RoBERTa/XLNet/BigBird/T5 "
                         "collapse to majority class on this small dataset.")
    ap.add_argument("--force", action="store_true", help="retrain even if model exists")
    ap.add_argument("--no-spacy", action="store_true")
    ap.add_argument("--spacy-epochs", type=int, default=20)
    args = ap.parse_args()

    zoo = bl.BERT_ZOO
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        zoo = [(n, hf) for (n, hf) in zoo if hf in want or n in want]

    summary = []
    for name, hf_id in zoo:
        out = bl.model_dir_for(hf_id)
        if (out / "best_model" / "config.json").exists() and not args.force:
            print(f"== skip {name} ({hf_id}) - already trained at {out}")
            summary.append((name, "skipped"))
            continue
        lr = 1e-4 if hf_id.split("/")[-1].startswith("t5") else args.lr
        print(f"\n== train {name} ({hf_id}) lr={lr} warmup={args.warmup} ==")
        cmd = [sys.executable, str(HERE / "train_verifier.py"), "--model", hf_id,
               "--epochs", str(args.epochs), "--batch", str(args.batch),
               "--lr", str(lr), "--warmup", str(args.warmup)]
        rc = subprocess.run(cmd).returncode
        summary.append((name, "ok" if rc == 0 else f"FAILED(rc={rc})"))

    if not args.no_spacy:
        sp = bl.MODELS / "verifier_spacy"
        if sp.exists() and not args.force:
            print(f"== skip spaCy baseline - already trained at {sp}")
            summary.append(("spacy", "skipped"))
        else:
            print("\n== train spaCy baseline ==")
            cmd = [sys.executable, str(HERE / "bench_spacy_train.py"),
                   "--epochs", str(args.spacy_epochs)]
            rc = subprocess.run(cmd).returncode
            summary.append(("spacy", "ok" if rc == 0 else f"FAILED(rc={rc})"))

    print("\n=== training summary ===")
    for name, status in summary:
        print(f"  {name:32s} {status}")
    print("\nNext: python bench_run.py --use-gliner --use-glirel")


if __name__ == "__main__":
    main()
