# CFLP Research Dataset Audit Report

## 1. Executive Summary
- **Dataset:** XES3G5M (Mathematics Online Learning Interaction Dataset)
- **Primary Domain:** Primary and Secondary Mathematics (Chinese Curriculum with English KC Translations)
- **Total Dataset Size:** 8.049 GB across 3881 files
- **Knowledge Components in Metadata:** 1175 (100% matched to English translated metadata)
- **Questions in Metadata:** 7652

## 2. Partition & Student Leakage Analysis
- **Train/Validation Partition:** 33,397 sequences across 14,453 unique students.
- **Cross-Validation Folds:** 5 balanced folds ({0: 6669, 1: 6669, 2: 6697, 3: 6677, 4: 6685}).
- **Independent Test Partition:** 3,613 sequences across 3,613 unique students.
- **Student Overlap between Train and Test:** **0** (Strictly disjoint; 0% leakage across split).

## 3. Sequence Encoding & Padding Mechanics
- **Max Padded Window:** 200 interaction steps.
- **Padding Representation:** `-1` values denote missing or inactive sequence slots.
- **Sequence Lengths (Active):** Min: 3, Max: 200, Mean: 153.88, Median: 200.0.
- **Valid Observed Responses:** Binary {0, 1}.

## 4. Concept Coverage
- **Total KCs Defined:** 1,175
- **KCs Observed in Interactions:** 865 (Active concepts with behavioral data).
- **Unobserved / Rare KCs:** 310 concepts present in curriculum taxonomy but not active in this sample.

*Generated deterministically by `scripts/audit_dataset.py`.*
