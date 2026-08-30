# Pipeline 1 - Verifier Benchmark

- **Stage 1&2:** GLiNER=real+dict-union | GLiREL=real+pairwise-union topk=3
- **Corpus:** tiered_test  (45 units, 248 intrinsic test rows)
- **Chain:** txt -> GLiNER -> GLiREL -> verifier -> SHACL -> KG

## Stage 3 - intrinsic VALID/INVALID (verifier_test)

| model | acc | macroF1 | validRec | rej:direction | rej:hard |
| --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 0.9234 | 0.9218 | 0.951 | 0.9362 | 0.8462 |
| distilbert_base_uncased | 0.8629 | 0.8566 | 0.7941 | 0.9681 | 0.8077 |
| xlnet_base_cased | 0.8306 | 0.8251 | 0.7941 | 0.9255 | 0.7308 |
| roberta_base | 0.629 | 0.6008 | 0.4412 | 0.6596 | 0.9423 |
| t5_base | 0.9315 | 0.9293 | 0.9216 | 0.9574 | 0.9038 |
| spacy | 0.8387 | 0.83 | 0.7451 | 0.9149 | 0.8846 |
| passthrough | 0.4113 | 0.2914 | 1.0 | 0.0 | 0.0 |

## End-to-end triple extraction (per tier)

### tier: `explicit`

| model | entRec(s1) | candCeil(s2) | P | R | F1 | avgCands |
| --- | --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 1.0 | 0.9091 | 0.8333 | 0.9091 | 0.8696 | 2.73 |
| distilbert_base_uncased | 1.0 | 0.9091 | 1.0 | 0.8182 | 0.9 | 2.73 |
| xlnet_base_cased | 1.0 | 0.9091 | 0.875 | 0.6364 | 0.7368 | 2.73 |
| roberta_base | 1.0 | 0.9091 | 0.5 | 0.2727 | 0.3529 | 2.73 |
| t5_base | 1.0 | 0.9091 | 1.0 | 0.9091 | 0.9524 | 2.73 |
| spacy | 1.0 | 0.9091 | 0.75 | 0.5455 | 0.6316 | 2.73 |
| passthrough | 1.0 | 0.9091 | 0.3333 | 0.9091 | 0.4878 | 2.73 |

### tier: `implicit`

| model | entRec(s1) | candCeil(s2) | P | R | F1 | avgCands |
| --- | --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 1.0 | 1.0 | 0.8947 | 1.0 | 0.9444 | 4.64 |
| distilbert_base_uncased | 1.0 | 1.0 | 0.8571 | 0.7059 | 0.7742 | 4.64 |
| xlnet_base_cased | 1.0 | 1.0 | 0.7778 | 0.8235 | 0.8 | 4.64 |
| roberta_base | 1.0 | 1.0 | 0.5 | 0.4706 | 0.4848 | 4.64 |
| t5_base | 1.0 | 1.0 | 0.8889 | 0.9412 | 0.9143 | 4.64 |
| spacy | 1.0 | 1.0 | 0.8125 | 0.7647 | 0.7879 | 4.64 |
| passthrough | 1.0 | 1.0 | 0.3333 | 1.0 | 0.5 | 4.64 |

### tier: `long_distance`

| model | entRec(s1) | candCeil(s2) | P | R | F1 | avgCands |
| --- | --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 3.27 |
| distilbert_base_uncased | 1.0 | 1.0 | 0.8889 | 0.7273 | 0.8 | 3.27 |
| xlnet_base_cased | 1.0 | 1.0 | 0.8333 | 0.9091 | 0.8696 | 3.27 |
| roberta_base | 1.0 | 1.0 | 0.4 | 0.3636 | 0.381 | 3.27 |
| t5_base | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 3.27 |
| spacy | 1.0 | 1.0 | 0.8182 | 0.8182 | 0.8182 | 3.27 |
| passthrough | 1.0 | 1.0 | 0.3056 | 1.0 | 0.4681 | 3.27 |

### tier: `nested`

| model | entRec(s1) | candCeil(s2) | P | R | F1 | avgCands |
| --- | --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 1.0 | 0.9818 | 0.3006 | 0.8909 | 0.4495 | 25.58 |
| distilbert_base_uncased | 1.0 | 0.9818 | 0.4468 | 0.7636 | 0.5638 | 25.58 |
| xlnet_base_cased | 1.0 | 0.9818 | 0.295 | 0.7455 | 0.4227 | 25.58 |
| roberta_base | 1.0 | 0.9818 | 0.2245 | 0.4 | 0.2876 | 25.58 |
| t5_base | 1.0 | 0.9818 | 0.5275 | 0.8727 | 0.6575 | 25.58 |
| spacy | 1.0 | 0.9818 | 0.2794 | 0.6909 | 0.3979 | 25.58 |
| passthrough | 1.0 | 0.9818 | 0.1759 | 0.9818 | 0.2983 | 25.58 |

### tier: `ALL`

| model | entRec(s1) | candCeil(s2) | P | R | F1 | avgCands |
| --- | --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 1.0 | 0.9787 | 0.4244 | 0.9255 | 0.5819 | 9.42 |
| distilbert_base_uncased | 1.0 | 0.9787 | 0.5635 | 0.7553 | 0.6455 | 9.42 |
| xlnet_base_cased | 1.0 | 0.9787 | 0.4068 | 0.766 | 0.5314 | 9.42 |
| roberta_base | 1.0 | 0.9787 | 0.2846 | 0.3936 | 0.3304 | 9.42 |
| t5_base | 1.0 | 0.9787 | 0.6538 | 0.9043 | 0.7589 | 9.42 |
| spacy | 1.0 | 0.9787 | 0.386 | 0.7021 | 0.4981 | 9.42 |
| passthrough | 1.0 | 0.9787 | 0.217 | 0.9787 | 0.3552 | 9.42 |

## Stage 4-5 - SHACL filter + knowledge graph (post-SHACL) vs expected

| model | F1_preSHACL | shaclRemoved | conforms | kg_P | kg_R | kg_F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bigbird_roberta_base | 0.568 | 0 | None | 0.408 | 0.9342 | 0.568 | 71 | 103 | 5 |
| distilbert_base_uncased | 0.6455 | 0 | None | 0.5398 | 0.8026 | 0.6455 | 61 | 52 | 15 |
| xlnet_base_cased | 0.5221 | 0 | None | 0.3933 | 0.7763 | 0.5221 | 59 | 91 | 17 |
| roberta_base | 0.3261 | 0 | None | 0.2778 | 0.3947 | 0.3261 | 30 | 78 | 46 |
| t5_base | 0.738 | 0 | None | 0.6216 | 0.9079 | 0.738 | 69 | 42 | 7 |
| spacy | 0.4843 | 0 | None | 0.3673 | 0.7105 | 0.4843 | 54 | 93 | 22 |
| passthrough | 0.3947 | 0 | None | 0.2467 | 0.9868 | 0.3947 | 75 | 229 | 1 |

---
_entRec = stage-1 entity recall; candCeil = stage-2 candidate recall (max achievable R for any verifier); P/R/F1 = post-verifier triples; `passthrough` = no verifier (accept all candidates)._

_Per-txt breakdown (one row per model x input text): `benchmark_per_txt_tiered_test.csv`._