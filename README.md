# Ontology-Driven RDF Triple Extraction — Caves of Qud

An information-extraction pipeline that populates a knowledge graph from natural
language text, using an OWL ontology as its schema. A BERT-family model acts as a
**VALID/INVALID verifier** of candidate triples rather than as a relation classifier,
so the system survives ontology changes without retraining.

```
text → GLiNER → GLiREL → verifier → SHACL → RDF triples
```

Each stage has a deterministic fallback, so the whole chain runs without the neural
models and upgrades by flipping a flag.

| Stage | Model | Fallback |
|:---|:---|:---|
| 1. Sentence split | masks aliases containing `.!?` | — |
| 2. Entity spans | GLiNER ∪ alias dictionary | exact alias matcher |
| 3. Relation candidates | GLiREL ∪ pairwise (top-k=3) | domain/range pairwise |
| 4. Verifier | BERT → VALID / INVALID | pass-through |
| 5. Constraints | SHACL (pyshacl) | domain/range enforced upstream |

## Why a verifier and not a classifier

A model that picks one of the 28 object properties is permanently bound to those 28
labels: changing the ontology means rebuilding the dataset and retraining. Here the
model receives an already-formed candidate triple plus the sentence that should
support it, and answers one binary question — *does this sentence support this
relation between these two entities?*

The input is marked with special tokens:

```
[REL] memberOf [/REL] [E1]Argyve[/E1] is a member of the [E2]Barathrumites[/E2].
```

## Ontology

`Final_Caves_of_QUD.owl` — 16 classes, 28 object properties (with inverses),
7 datatype properties, 63 individuals. The ABox is already populated, which makes
supervision free: **263 gold triples** (208 object + 55 datatype) are read directly
from the asserted facts, with no manual annotation.

## Results

Benchmark on `tiered_test` (45 texts, 248 verifier rows), stages 1 & 2 running the
real GLiNER + GLiREL models in union with their fallbacks.

| Model | Verifier F1 | KG Precision | KG Recall | KG F1 |
|:---|:---|:---|:---|:---|
| **t5_base** | **91.7%** | 62.2% | 90.8% | **73.8%** |
| distilbert_base_uncased | 82.7% | 54.0% | 80.3% | 64.5% |
| bigbird_roberta_base | 91.1% | 40.8% | 93.4% | 56.8% |
| xlnet_base_cased | 79.4% | 39.3% | 77.6% | 52.2% |
| spaCy (baseline) | 79.2% | 36.7% | 71.0% | 48.4% |
| roberta_base | 49.5% | 27.8% | 39.5% | 32.6% |
| _passthrough (no verifier)_ | — | 24.7% | 98.7% | 39.5% |

The `passthrough` row accepts every candidate and is the pipeline's precision floor;
the jump from 24.7% to 62.2% precision is the verifier's contribution.

End-to-end F1 by difficulty tier:

| Model | explicit | implicit | long_distance | nested |
|:---|:---|:---|:---|:---|
| t5_base | 95.2% | 91.4% | 100% | 65.8% |
| distilbert_base_uncased | 90.0% | 77.4% | 80.0% | 56.4% |
| bigbird_roberta_base | 87.0% | 94.4% | 100% | 45.0% |

`nested` (high triple density per sentence) is the current bottleneck; `roberta_base`
is under-trained and needs a 40-epoch run.

## Layout

```
Final_Caves_of_QUD.owl        the ontology (schema + populated ABox)
pipeline1/
  scripts/                    pipeline, dataset builders, training, benchmark
  data-input/{train,val,test} 300 generated stories
  generated/                  ontology interface, gold triples, tiered corpus,
                              verifier splits
  outputs/                    benchmark reports, per-text metrics, KG graphs
  models/                     trained weights (gitignored — rebuild locally)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Trained weights are not in the repository (2.5 GB; individual files exceed GitHub's
100 MB limit). Rebuild them with `bench_train_all.py` below.

## Running

Regenerate the schema and datasets from the ontology:

```bash
cd pipeline1/scripts
python build_qud_dataset.py
python build_gliner_glirel_config.py
python gen_tiered_expand.py
python split_tiered.py
python build_verifier_from_tiered.py
```

Train the verifiers (`roberta`, `bigbird` and `t5` need `--epochs 40`):

```bash
python bench_train_all.py
```

Run the benchmark and build the full report:

```bash
python bench_run.py --use-gliner --use-glirel
python bench_report.py
```

Extract triples from a single text:

```bash
python infer_pipeline.py --file ../data-input/test/story_010.txt \
    --use-gliner --use-glirel --model ../models/verifier_t5_base/best_model
```

## Outputs

| File | Contents |
|:---|:---|
| `benchmark_full_report.md` | per-story metrics, model comparison, confusion matrices, dataset splits, ontology tables |
| `benchmark_prf.md` | slim precision/recall/F1 comparison |
| `benchmark_per_txt_tiered_test.csv` | one row per model × text |
| `verifier_splits_eval.json` | cached predictions for train/val/test |
| `kg_<model>_predicted.ttl` | assembled knowledge graph per verifier |
| `kg_<model>_diff.tsv` | TP / FP / FN against `kg_expected.ttl` |

## Training notes

The dataset is small (1042 training rows), so fine-tuning is unstable: RoBERTa,
XLNet and BigBird collapse to the majority class without warmup and at lr ≥ 3e-5.
Stable recipe — **lr 1e-5, batch 8, warmup 0.1**. BigBird and T5 converge only around
epoch 17–20 and need 40 epochs; T5 uses lr 1e-4 for its fresh head. DistilBERT is
robust to all settings.
