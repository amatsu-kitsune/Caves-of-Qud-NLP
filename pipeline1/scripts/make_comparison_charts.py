"""
Build transformer-comparison charts from the LAST training + benchmark results.
Reads : outputs/benchmark_results.json, outputs/training_log.csv
Writes: outputs/charts/*.png  (+ a combined dashboard)

Run:  python make_comparison_charts.py
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

PARAMS_M = {"distilbert_base_uncased": 66, "xlnet_base_cased": 110, "roberta_base": 125,
            "bigbird_roberta_base": 127, "t5_base": 223, "spacy": 1}
NICE = {"distilbert_base_uncased": "DistilBERT", "xlnet_base_cased": "XLNet",
        "roberta_base": "RoBERTa", "bigbird_roberta_base": "BigBird",
        "t5_base": "T5", "spacy": "spaCy", "passthrough": "passthrough"}
COL = {"distilbert_base_uncased": "#2ca02c", "xlnet_base_cased": "#1f77b4",
       "roberta_base": "#9467bd", "bigbird_roberta_base": "#d62728",
       "t5_base": "#ff7f0e", "spacy": "#7f7f7f", "passthrough": "#bcbcbc"}
TRANSFORMERS = ["distilbert_base_uncased", "xlnet_base_cased", "roberta_base",
                "bigbird_roberta_base", "t5_base"]

r = json.load(open(OUT / "benchmark_results.json", encoding="utf-8"))
M = r["models"]

def intr(n, k): return M[n]["intrinsic"][k]
def e2e_all(n, k): return M[n].get("e2e", {}).get("per_tier", {}).get("ALL", {}).get(k)

order = sorted([n for n in M if "intrinsic" in M[n]], key=lambda n: -intr(n, "macro_f1"))

rows = list(csv.DictReader(open(OUT / "training_log.csv", encoding="utf-8")))
runs = defaultdict(list); cur = defaultdict(list)
for x in rows:
    enc = x["encoder"]; ep = int(x["epoch"])
    if ep == 1 and cur[enc]:
        runs[enc].append(cur[enc]); cur[enc] = []
    cur[enc].append(float(x["val_macro_f1"]))
for enc in cur:
    if cur[enc]: runs[enc].append(cur[enc])
ENC_KEY = {"distilbert-base-uncased": "distilbert_base_uncased", "xlnet-base-cased": "xlnet_base_cased",
           "roberta-base": "roberta_base", "google/bigbird-roberta-base": "bigbird_roberta_base",
           "t5-base": "t5_base", "spacy-textcat": "spacy"}
curves = {ENC_KEY[e]: v[-1] for e, v in runs.items() if e in ENC_KEY}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "figure.autolayout": False})
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
fig.suptitle(f"Stage-3 Verifier — Transformer comparison  ·  corpus `{r['corpus']}`: "
             f"{r['n_test_rows']} rows intrinsic / {r['n_units']} units e2e\n"
             f"stages 1&2: {r['stage12']}",
             fontsize=13, fontweight="bold")

a = ax[0, 0]
for n in TRANSFORMERS + ["spacy"]:
    x, y = PARAMS_M[n], intr(n, "macro_f1")
    a.scatter(x, y, s=90 + 500 * intr(n, "accuracy"), color=COL[n], edgecolor="k",
              alpha=.8, zorder=3)
    a.annotate(NICE[n], (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
a.set_xlabel("model size  (M params  →  less efficient)")
a.set_ylabel("test macro-F1")
a.set_title("A · Accuracy vs efficiency  (bubble = accuracy)")
a.text(.98, .03, "↖ better", transform=a.transAxes, ha="right", fontsize=11, color="#555")

b = ax[0, 1]
xs = range(len(order)); w = .4
b.bar([i - w/2 for i in xs], [intr(n, "macro_f1") for n in order], w, label="macro-F1",
      color=[COL[n] for n in order])
b.bar([i + w/2 for i in xs], [intr(n, "accuracy") for n in order], w, label="accuracy",
      color=[COL[n] for n in order], alpha=.5, hatch="//")
b.set_xticks(list(xs)); b.set_xticklabels([NICE[n] for n in order], rotation=35, ha="right")
b.set_ylabel("score"); b.set_ylim(0, 1.05)
b.set_title("B · Verifier classification (VALID/INVALID, test)"); b.legend(fontsize=8)

c = ax[0, 2]
for n in order:
    it = M[n]["intrinsic"]; rj = it["reject_recall"]
    meanrej = sum(rj.values()) / len(rj) if rj else 0.0
    c.scatter(meanrej, it["valid_recall"], s=140, color=COL[n], edgecolor="k", zorder=3)
    c.annotate(NICE[n], (meanrej, it["valid_recall"]), xytext=(5, 4),
               textcoords="offset points", fontsize=8)
c.plot([0, 1], [1, 0], "k--", alpha=.3)
c.set_xlabel("reject-recall  (catches bad triples →)"); c.set_ylabel("valid-recall  (keeps good triples ↑)")
c.set_xlim(-.05, 1.05); c.set_ylim(-.05, 1.05)
c.set_title("C · Accept vs reject balance  (↗ better)")

d = ax[1, 0]
for n in TRANSFORMERS + ["spacy"]:
    if n in curves:
        d.plot(range(1, len(curves[n]) + 1), curves[n], marker="o", ms=3,
                color=COL[n], label=NICE[n], lw=1.6)
d.axhline(0.37, ls=":", color="k", alpha=.4)
d.text(1, .385, "majority-class collapse (~0.37)", fontsize=7, color="#555")
d.set_xlabel("epoch"); d.set_ylabel("val macro-F1"); d.set_ylim(0, 1.05)
d.set_title("D · Training convergence (validation, last run)"); d.legend(fontsize=8, ncol=2)

e = ax[1, 1]
oe = [n for n in order if e2e_all(n, "f1") is not None]
e.bar(range(len(oe)), [e2e_all(n, "f1") for n in oe], color=[COL[n] for n in oe])
for i, n in enumerate(oe):
    e.text(i, e2e_all(n, "f1") + .01, f"{e2e_all(n,'f1'):.2f}", ha="center", fontsize=8)
e.set_ylabel("triple micro-F1"); e.set_ylim(0, max(.5, max(e2e_all(n, "f1") for n in oe) + .1))
e.set_xticks(range(len(oe))); e.set_xticklabels([NICE[n] for n in oe], rotation=35, ha="right")
e.set_title(f"E · End-to-end triples ({r['n_units']} units)")

f = ax[1, 2]
tiers = ["explicit", "implicit", "long_distance", "nested"]
ranked = sorted((n for n in TRANSFORMERS if e2e_all(n, "f1") is not None),
                key=lambda n: -e2e_all(n, "f1"))
show = ranked[:3] + (["spacy"] if e2e_all("spacy", "f1") is not None else [])
x = range(len(tiers)); w = .8 / max(1, len(show))
for j, n in enumerate(show):
    pt = M[n]["e2e"]["per_tier"]
    f.bar([i + (j - (len(show) - 1) / 2) * w for i in x],
          [pt.get(t, {}).get("f1", 0) for t in tiers], w,
          color=COL[n], label=NICE[n])
f.set_xticks(list(x)); f.set_xticklabels([t.replace("_", "\n") for t in tiers], fontsize=8)
f.set_ylabel("triple F1"); f.set_ylim(0, 1.05)
f.set_title("F · e2e F1 by tier  (top-3 + baseline)"); f.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(CH / "transformer_comparison.png", dpi=150)
print("wrote", CH / "transformer_comparison.png")

summary = {n: {"params_M": PARAMS_M.get(n), "test_accuracy": intr(n, "accuracy"),
               "test_macro_f1": intr(n, "macro_f1"), "valid_recall": M[n]["intrinsic"]["valid_recall"],
               "e2e_f1": e2e_all(n, "f1")} for n in order}
(CH / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", CH / "comparison_summary.json")
