# Benchmark - spaCy baseline vs Transformers (P / R / F1)

- corpus (end-to-end): `tiered_test`  |  stages 1&2: GLiNER=real+dict-union | GLiREL=real+pairwise-union topk=3
- verifier test rows: 248  |  e2e units: 45

## A. Verifier classification - VALID/INVALID (macro, on verifier_test)

| model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| spaCy (baseline) | 0.8399 | 0.8246 | 0.83 |
| bigbird_roberta_base | 0.9187 | 0.9275 | 0.9218 |
| distilbert_base_uncased | 0.8627 | 0.8525 | 0.8566 |
| xlnet_base_cased | 0.8251 | 0.8251 | 0.8251 |
| roberta_base | 0.6116 | 0.6007 | 0.6008 |
| t5_base | 0.9287 | 0.93 | 0.9293 |

## B. Knowledge graph - post-SHACL, deduplicated (predicted vs expected)

| model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| spaCy (baseline) | 0.3673 | 0.7105 | 0.4843 |
| bigbird_roberta_base | 0.408 | 0.9342 | 0.568 |
| distilbert_base_uncased | 0.5398 | 0.8026 | 0.6455 |
| xlnet_base_cased | 0.3933 | 0.7763 | 0.5221 |
| roberta_base | 0.2778 | 0.3947 | 0.3261 |
| t5_base | 0.6216 | 0.9079 | 0.738 |

_per-model graphs in `kg_<model>_predicted.ttl` vs `kg_expected.ttl`; TP/FP/FN in `kg_<model>_diff.tsv` and `kg_<model>_labeled.ttl`._
