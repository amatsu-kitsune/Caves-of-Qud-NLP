# Pipeline 1 - Full Benchmark Report

- **Chain:** txt -> GLiNER -> GLiREL -> verifier -> SHACL -> KG
- **Stages 1&2:** GLiNER=real+dict-union | GLiREL=real+pairwise-union topk=3
- **End-to-end corpus:** `tiered_test` (45 texts)
- **Verifier rows:** train 1042 · val 217 · test 248

## 1. Per-story performance (TP / FP / FN)

### spaCy (baseline)

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_008 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_010 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_013 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_020 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_031 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_035 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_041 | implicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_045 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_047 | implicit | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_051 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_052 | long_distance | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_069 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_070 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_074 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_078 | nested | 3 | 27 | 4 | 10.0% | 42.9% | 16.2% |
| qt_079 | nested | 4 | 8 | 2 | 33.3% | 66.7% | 44.4% |
| qt_094 | nested | 6 | 3 | 0 | 66.7% | 100.0% | 80.0% |
| qt_096 | nested | 3 | 12 | 2 | 20.0% | 60.0% | 30.0% |
| qt_103 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_117 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_122 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_144 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| **TOTAL (micro)** | 45 txt | **66** | **105** | **28** | **38.6%** | **70.2%** | **49.8%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

### bigbird_roberta_base

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_008 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_010 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_013 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_020 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_031 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_035 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_041 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_045 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_047 | implicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_051 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_052 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_069 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_070 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_074 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_078 | nested | 6 | 26 | 1 | 18.8% | 85.7% | 30.8% |
| qt_079 | nested | 4 | 6 | 2 | 40.0% | 66.7% | 50.0% |
| qt_094 | nested | 5 | 3 | 1 | 62.5% | 83.3% | 71.4% |
| qt_096 | nested | 5 | 20 | 0 | 20.0% | 100.0% | 33.3% |
| qt_103 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_117 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_122 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_144 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| **TOTAL (micro)** | 45 txt | **87** | **118** | **7** | **42.4%** | **92.6%** | **58.2%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

### distilbert_base_uncased

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_008 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_010 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_013 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_020 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_031 | implicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_035 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_041 | implicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_045 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_047 | implicit | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_051 | long_distance | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_052 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_069 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_070 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_074 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_078 | nested | 5 | 23 | 2 | 17.9% | 71.4% | 28.6% |
| qt_079 | nested | 1 | 4 | 5 | 20.0% | 16.7% | 18.2% |
| qt_094 | nested | 5 | 0 | 1 | 100.0% | 83.3% | 90.9% |
| qt_096 | nested | 3 | 2 | 2 | 60.0% | 60.0% | 60.0% |
| qt_103 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_117 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_122 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_144 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| **TOTAL (micro)** | 45 txt | **71** | **55** | **23** | **56.3%** | **75.5%** | **64.5%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

### xlnet_base_cased

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_008 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_010 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_013 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_020 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_031 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_035 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_041 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_045 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_047 | implicit | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_051 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_052 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_069 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_070 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_074 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_078 | nested | 6 | 23 | 1 | 20.7% | 85.7% | 33.3% |
| qt_079 | nested | 2 | 7 | 4 | 22.2% | 33.3% | 26.7% |
| qt_094 | nested | 6 | 4 | 0 | 60.0% | 100.0% | 75.0% |
| qt_096 | nested | 3 | 10 | 2 | 23.1% | 60.0% | 33.3% |
| qt_103 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_117 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_122 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_144 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| **TOTAL (micro)** | 45 txt | **72** | **105** | **22** | **40.7%** | **76.6%** | **53.1%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

### roberta_base

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_008 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_010 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_013 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_020 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_031 | implicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_035 | implicit | 2 | 2 | 0 | 50.0% | 100.0% | 66.7% |
| qt_041 | implicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_045 | implicit | 1 | 1 | 1 | 50.0% | 50.0% | 50.0% |
| qt_047 | implicit | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_051 | long_distance | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_052 | long_distance | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_069 | long_distance | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_070 | long_distance | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_074 | long_distance | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_078 | nested | 2 | 22 | 5 | 8.3% | 28.6% | 12.9% |
| qt_079 | nested | 2 | 6 | 4 | 25.0% | 33.3% | 28.6% |
| qt_094 | nested | 4 | 3 | 2 | 57.1% | 66.7% | 61.5% |
| qt_096 | nested | 1 | 9 | 4 | 10.0% | 20.0% | 13.3% |
| qt_103 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_117 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_122 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_144 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| **TOTAL (micro)** | 45 txt | **37** | **93** | **57** | **28.5%** | **39.4%** | **33.0%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

### t5_base

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_008 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_010 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_013 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_020 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_031 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_035 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_041 | implicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_045 | implicit | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_047 | implicit | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| qt_051 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_052 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_069 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_070 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_074 | long_distance | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_078 | nested | 7 | 7 | 0 | 50.0% | 100.0% | 66.7% |
| qt_079 | nested | 2 | 4 | 4 | 33.3% | 33.3% | 33.3% |
| qt_094 | nested | 5 | 2 | 1 | 71.4% | 83.3% | 76.9% |
| qt_096 | nested | 4 | 9 | 1 | 30.8% | 80.0% | 44.4% |
| qt_103 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_117 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_122 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| qt_144 | explicit | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| **TOTAL (micro)** | 45 txt | **85** | **45** | **9** | **65.4%** | **90.4%** | **75.9%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

### passthrough

| Story | Tier | TP | FP | FN | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| qt_007 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_008 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_010 | explicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_013 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_020 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_031 | implicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_035 | implicit | 2 | 2 | 0 | 50.0% | 100.0% | 66.7% |
| qt_041 | implicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_045 | implicit | 2 | 6 | 0 | 25.0% | 100.0% | 40.0% |
| qt_047 | implicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_051 | long_distance | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_052 | long_distance | 1 | 5 | 0 | 16.7% | 100.0% | 28.6% |
| qt_069 | long_distance | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_070 | long_distance | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_074 | long_distance | 1 | 5 | 0 | 16.7% | 100.0% | 28.6% |
| qt_078 | nested | 7 | 47 | 0 | 13.0% | 100.0% | 22.9% |
| qt_079 | nested | 6 | 16 | 0 | 27.3% | 100.0% | 42.9% |
| qt_094 | nested | 6 | 14 | 0 | 30.0% | 100.0% | 46.2% |
| qt_096 | nested | 5 | 31 | 0 | 13.9% | 100.0% | 24.4% |
| qt_103 | explicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_117 | explicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_122 | explicit | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| qt_126 | explicit | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| qt_140 | explicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| qt_144 | explicit | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| **TOTAL (micro)** | 45 txt | **92** | **332** | **2** | **21.7%** | **97.9%** | **35.5%** |

_showing first 25 of 45 texts; full data in `benchmark_per_txt_tiered_test.csv`. TOTAL row covers all 45._

## 2. Model comparison summary

Micro-averaged over the whole evaluation corpus (`tiered_test`, 45 texts).

| Model | Precision | Recall | F1-score | KG Precision | KG Recall | KG F1 |
|:---|:---|:---|:---|:---|:---|:---|
| spaCy (baseline) | 38.6% | 70.2% | 49.8% | 36.7% | 71.0% | 48.4% |
| bigbird_roberta_base | 42.4% | 92.5% | 58.2% | 40.8% | 93.4% | 56.8% |
| distilbert_base_uncased | 56.4% | 75.5% | 64.5% | 54.0% | 80.3% | 64.5% |
| xlnet_base_cased | 40.7% | 76.6% | 53.1% | 39.3% | 77.6% | 52.2% |
| roberta_base | 28.5% | 39.4% | 33.0% | 27.8% | 39.5% | 32.6% |
| t5_base | 65.4% | 90.4% | 75.9% | 62.2% | 90.8% | 73.8% |
| passthrough | 21.7% | 97.9% | 35.5% | 24.7% | 98.7% | 39.5% |

Per-tier breakdown of the end-to-end F1:

| Model | explicit | implicit | long_distance | nested | ALL |
|:---|:---|:---|:---|:---|:---|
| spaCy (baseline) | 63.2% | 78.8% | 81.8% | 39.8% | 49.8% |
| bigbird_roberta_base | 87.0% | 94.4% | 100.0% | 45.0% | 58.2% |
| distilbert_base_uncased | 90.0% | 77.4% | 80.0% | 56.4% | 64.5% |
| xlnet_base_cased | 73.7% | 80.0% | 87.0% | 42.3% | 53.1% |
| roberta_base | 35.3% | 48.5% | 38.1% | 28.8% | 33.0% |
| t5_base | 95.2% | 91.4% | 100.0% | 65.8% | 75.9% |
| passthrough | 48.8% | 50.0% | 46.8% | 29.8% | 35.5% |

## 3. Verifier confusion matrices

### spaCy (baseline)

**Train** — 1042 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 572 | 29 |
| **Actual VALID** | 58 | 383 |

**Val** — 217 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 118 | 11 |
| **Actual VALID** | 19 | 69 |

**Test** — 248 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 132 | 14 |
| **Actual VALID** | 26 | 76 |

### bigbird_roberta_base

**Train** — 1042 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 553 | 48 |
| **Actual VALID** | 3 | 438 |

**Val** — 217 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 119 | 10 |
| **Actual VALID** | 6 | 82 |

**Test** — 248 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 132 | 14 |
| **Actual VALID** | 5 | 97 |

### distilbert_base_uncased

**Train** — 1042 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 589 | 12 |
| **Actual VALID** | 44 | 397 |

**Val** — 217 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 121 | 8 |
| **Actual VALID** | 17 | 71 |

**Test** — 248 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 133 | 13 |
| **Actual VALID** | 21 | 81 |

### xlnet_base_cased

**Train** — 1042 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 532 | 69 |
| **Actual VALID** | 81 | 360 |

**Val** — 217 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 116 | 13 |
| **Actual VALID** | 20 | 68 |

**Test** — 248 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 125 | 21 |
| **Actual VALID** | 21 | 81 |

### roberta_base

**Train** — 1042 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 398 | 203 |
| **Actual VALID** | 170 | 271 |

**Val** — 217 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 95 | 34 |
| **Actual VALID** | 41 | 47 |

**Test** — 248 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 111 | 35 |
| **Actual VALID** | 57 | 45 |

### t5_base

**Train** — 1042 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 584 | 17 |
| **Actual VALID** | 13 | 428 |

**Val** — 217 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 121 | 8 |
| **Actual VALID** | 4 | 84 |

**Test** — 248 rows

|  | Pred. INVALID | Pred. VALID |
|:---|:---|:---|
| **Actual INVALID** | 137 | 9 |
| **Actual VALID** | 8 | 94 |

## 4. Dataset split

| Split | Rows | NEG (INVALID) | POS (VALID) | involvesFinding | isGivenBy | isHabitantOf | memberOf | hasRace | isSearchTargetIn | giveQuest | dislikedBy |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 601 | 441 | 72 | 53 | 60 | 47 | 46 | 43 | 41 | 43 |
| Val | 217 | 129 | 88 | 18 | 20 | 8 | 12 | 8 | 15 | 8 | 11 |
| Test | 248 | 146 | 102 | 19 | 13 | 16 | 16 | 20 | 15 | 20 | 13 |

_columns after POS are the 8 most frequent candidate relations of 33 distinct._

Negative examples by construction type:

| Split | neg: direction | neg: dr | neg: hard | object | datatype |
|:---|:---|:---|:---|:---|:---|
| Train | 395 | 13 | 193 | 976 | 66 |
| Val | 78 | 4 | 47 | 205 | 12 |
| Test | 94 | 0 | 52 | 232 | 16 |

## 5. Verifier summary per split

Positive class = `VALID`; macro-F1 averages both classes.

### spaCy (baseline)

| Split | Rows | Accuracy | Recall | Precision | F1 | Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 91.6% | 86.9% | 93.0% | 89.8% | 91.4% |
| Val | 217 | 86.2% | 78.4% | 86.2% | 82.1% | 85.4% |
| Test | 248 | 83.9% | 74.5% | 84.4% | 79.2% | 83.0% |

### bigbird_roberta_base

| Split | Rows | Accuracy | Recall | Precision | F1 | Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 95.1% | 99.3% | 90.1% | 94.5% | 95.0% |
| Val | 217 | 92.6% | 93.2% | 89.1% | 91.1% | 92.4% |
| Test | 248 | 92.3% | 95.1% | 87.4% | 91.1% | 92.2% |

### distilbert_base_uncased

| Split | Rows | Accuracy | Recall | Precision | F1 | Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 94.6% | 90.0% | 97.1% | 93.4% | 94.4% |
| Val | 217 | 88.5% | 80.7% | 89.9% | 85.0% | 87.8% |
| Test | 248 | 86.3% | 79.4% | 86.2% | 82.7% | 85.7% |

### xlnet_base_cased

| Split | Rows | Accuracy | Recall | Precision | F1 | Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 85.6% | 81.6% | 83.9% | 82.8% | 85.2% |
| Val | 217 | 84.8% | 77.3% | 84.0% | 80.5% | 84.0% |
| Test | 248 | 83.1% | 79.4% | 79.4% | 79.4% | 82.5% |

### roberta_base

| Split | Rows | Accuracy | Recall | Precision | F1 | Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 64.2% | 61.5% | 57.2% | 59.2% | 63.7% |
| Val | 217 | 65.4% | 53.4% | 58.0% | 55.6% | 63.7% |
| Test | 248 | 62.9% | 44.1% | 56.2% | 49.5% | 60.1% |

### t5_base

| Split | Rows | Accuracy | Recall | Precision | F1 | Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|
| Train | 1042 | 97.1% | 97.0% | 96.2% | 96.6% | 97.0% |
| Val | 217 | 94.5% | 95.5% | 91.3% | 93.3% | 94.3% |
| Test | 248 | 93.2% | 92.2% | 91.3% | 91.7% | 92.9% |

## 6. Ontology structure

**Class hierarchy**

| Core class | Subclasses |
|:---|:---|
| Character | Mutant, NPC, Organic, PlayerCharacter, Robot, Truekin |
| Faction | - |
| Item | - |
| Place | - |
| Quest | - |
| Race | - |
| Role | Fighter, Trader |
| Settlement | - |

**Object properties**

| Property | Domain | Range | Inverse |
|:---|:---|:---|:---|
| containsSettlement | Place | Settlement | locatedInPlace |
| dislikedBy | Character | Faction | dislikes |
| dislikes | Faction | Character | dislikedBy |
| foughtIn | Faction | Quest | involvesFightWith |
| giveQuest | NPC | Quest | isGivenBy |
| hasIndividual | Race | Mutant | hasRace |
| hasInhabitant | Place ⊔ Settlement | NPC | isHabitantOf |
| hasMember | Faction | NPC | memberOf |
| hasRace | Mutant | Race | hasIndividual |
| hasRole | NPC | Role | isRoleOf |
| hates | Faction | Faction | - |
| hostsQuest | Place ⊔ Settlement | Quest | takesPlaceIn |
| involvesFightWith | Quest | Faction | foughtIn |
| involvesFinding | Quest | Item ⊔ NPC | isSearchTargetIn |
| isGivenBy | Quest | NPC | giveQuest |
| isHabitantOf | NPC | Place ⊔ Settlement | hasInhabitant |
| isRewardFor | Item | Quest | rewardsWith |
| isRoleOf | Role | NPC | hasRole |
| isSearchTargetIn | Item ⊔ NPC | Quest | involvesFinding |
| likedBy | Character | Faction | likes |
| likes | Faction | Character | likedBy |
| locatedInPlace | Settlement | Place | containsSettlement |
| memberOf | NPC | Faction | hasMember |
| requiresCompleting | Quest | Quest | - |
| rewardsWith | Quest | Item | isRewardFor |
| sellsItem | Trader | Item | - |
| spawnsIn | PlayerCharacter | Place | - |
| takesPlaceIn | Quest | Place ⊔ Settlement | hostsQuest |

**Datatype properties**

| Property | Domain | Range |
|:---|:---|:---|
| Boss | Fighter | boolean |
| HP | - | - |
| Tier | Trader | - |
| Value | - | - |
| Weight | - | - |
| level | Character | integer |
| strata | Place ⊔ Settlement | integer |

**Individuals per class**

| Class | N | Individuals |
|:---|:---|:---|
| Faction | 6 | Barathrumites, Girsh, Mechanimists, Putus Templar, Seraphic Covenant, Villagers of Joppa |
| Fighter | 2 | Fighter Boss, Fighter NonBoss |
| Item | 7 | Copper Wire, Joppa Recoiler, Scrapped Waydroid, Sparafucile s Carbine, metamorphic polygel, neutron flux, quantum mote |
| Mutant | 11 | Argyve, Asphodel Earl of Omonporch, Barathrum the Old, Golem, Otho, Pax Qlanq, Phinae Hoshaiah High Priest of the Rock, Slog of the Cloaca, Sparafucile, Starformed Ehalcodon, alchemist |
| NPC | 15 | Argyve, Asphodel Earl of Omonporch, Baetyl, Barathrum the Old, Golem, Herodododicus, Otho, Pax Qlanq, Phinae Hoshaiah High Priest of the Rock, Reseph, Saad Amus, Slog of the Cloaca, Sparafucile, Starformed Ehalcodon, alchemist |
| Place | 7 | Bethesda Susa, Eaters Tomb, Golgotha, Omonporch, Rainbow Wood, Rust Wells, Starfarers Quay |
| PlayerCharacter | 1 | Kun |
| Quest | 11 | A Call to Arms, A Canticle for Barathrum, Decoding the Signal, More Than a Willing Spirit, Pax Qlanq I Presume?, Reclamation, The Earl of Omonporch, The Golem, Tomb of the Eaters, We Are Starfreight, Weirdwire Conduit Eureka! |
| Race | 8 | Fungi, Girsh Nephilim, Mollusk, Mutated Human, Robot, Sentient Plant, Truekin, Urshiib |
| Robot | 3 | Baetyl, Herodododicus, Reseph |
| Settlement | 3 | Grit Gate, Joppa, Temple of the Rock |
| Trader | 3 | Trader high, Trader low, Trader mid |
| Truekin | 2 | Kun, Saad Amus |
