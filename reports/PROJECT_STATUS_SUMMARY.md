# C-PrIM Repository Status & Authoritative Evidence Summary

> [!NOTE]
> **Audit Note (Corrected on 2026-11-12):** Corrected validation-selected threshold $\tau^*$ from $0.7786$ to $0.8256$ and GNN test PR-AUC from $0.8872$ to $0.8775$ to match `reports/evidence_final_20261111/metrics_by_model.csv` exactly.

**Audit Date:** November 12, 2026  
**Canonical Evidence Directory:** [`reports/evidence_final_20261111/`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/evidence_final_20261111/)  
**Canonical Configuration:** `embedding_dim=64`, `hidden_dim=64`, `epochs=12`, `lr=0.003`, `weight_decay=1e-4`, `batch_size=64`, `seed=42` (with multi-seed robustness across seeds `{42, 123, 999}`).  
**Automated Test Suite Status:** **11 / 11 PASSED** (100% test pass rate recorded in [`reports/TEST_RUN_FINAL.txt`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/TEST_RUN_FINAL.txt)).  
**Quickstart & Service Verification:** **CONFIRMED** (`pytest -v` clean pass; `uvicorn cflp.api.main:app` healthy with 1,175 KCs and 1,062 graph edges).

---

## 1. Authoritative vs. Superseded Classification of `reports/`

Every file and directory in `reports/` is categorized below into its operational role for the IEEE manuscript update:

### A. Authoritative Research Deliverables (Source of Truth for Paper)
The following files are **strictly synchronized** with the canonical run (`evidence_final_20261111`) and represent the authoritative source of truth for manuscript numbers, tables, figures, and citations:

- **[`reports/RESULTS_READY_FOR_PAPER.md`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/RESULTS_READY_FOR_PAPER.md)**: **PRIMARY AUTHORITATIVE SUMMARY**. One-page master document presenting Table I equivalent (Predictive Performance), Table II equivalent (Ablation), Table III equivalent (Multi-Seed Robustness), Sensitivity Analysis, Literature Benchmark, and Path Divergence Statistic.
- **[`reports/FIGURE_MANIFEST.md`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/FIGURE_MANIFEST.md)**: Master figure inventory indexing all 11 figures with one-sentence IEEE paper captions, manuscript section mappings, and curated 4-figure / 5-figure subsets.
- **[`reports/figures_final/`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/figures_final/)**: The complete consolidated set of 11 publication-grade figures at 300 DPI in both vector (**PDF**) and raster (**PNG**) formats (22 files total).
- **[`reports/tables/`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/tables/)**: Authoritative CSV and LaTeX tables generated from the canonical run:
  - `table1_dataset_statistics.csv`: Dataset attributes and sequence distributions.
  - `table2_experimental_configuration.csv`: Canonical hardware, environment, and hyperparameter configuration.
  - `table3_test_performance_comparison.csv`: Test performance across models and threshold regimes ($\tau^* = 0.8256$ vs $\tau = 0.50$).
  - `table4_counterfactual_prerequisite_ranking.csv`: Prerequisite candidate rankings for target KC 5 under $\theta = 0.85$.
  - `table5_learning_path_comparison.csv`: Step-by-step path comparison (Proposed C-PrIM vs Weakest-Mastery-First).
  - `table6_ablation_study.csv`: Architecture ablation (Full GNN vs No-Graph MLP vs Baseline Logistic).
  - `table7_multi_seed_robustness.csv`: Independent multi-seed evaluation across seeds 42, 123, 999 ($0.7073 \pm 0.0104$).
  - `table_sensitivity_analysis.csv`: Delta-sensitivity analysis across intervention levels $\theta \in [0.60, 0.75, 0.85, 0.95]$.
  - `table_literature_benchmark_FINAL.csv`: Clean, IEEE-table-ready literature comparison.
  - `ieee_tables.tex`: LaTeX-ready code for Table A and Table B.
- **[`reports/CANONICAL_RUN_RECONCILIATION.md`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/CANONICAL_RUN_RECONCILIATION.md)**: Comprehensive audit reconciling earlier runs with the canonical run.
- **[`reports/IEEE_COMPARATIVE_EVALUATION.md`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/IEEE_COMPARATIVE_EVALUATION.md)**: Deep comparative analysis between sequential DLKT baselines (DKT, SAKT, GKT, AKT, LKT) and C-PrIM.
- **[`reports/REFERENCES_VERIFIED.md`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/REFERENCES_VERIFIED.md)**: IEEE bibliographic entries and verification audit for manuscript citations `[1]` through `[14]`.
- **[`reports/TEST_RUN_FINAL.txt`](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/TEST_RUN_FINAL.txt)**: Verifiable log of the automated test run (11 passed).

---

### B. Superseded Evidence Directories & Tables (Historical Archives)
The following directories and files reflect older exploratory runs, prior configurations (e.g., dim 32, 5 epochs), or earlier uncalibrated thresholds and must **not** be used for manuscript updates:

- **`reports/evidence_rerun_20261022_050835/`**: **SUPERSEDED**. Preceding evidence package from October 2026. Fully superseded by `evidence_final_20261111/`.
- **`reports/reports_output_sync/`**: **SUPERSEDED**. Intermediate synchronization snapshot. Fully superseded by `evidence_final_20261111/` and `reports/figures_final/`.
- **`reports/figures/` (loose root files)**: **SUPERSEDED / PROVENANCE ONLY**. Figures in `reports/figures_final/` represent the unified, confirmed 300 DPI publication package.
- **`reports/tables/table_counterfactual_rankings.csv` & `table_learning_path_comparison.csv`**: **SUPERSEDED / COMPATIBILITY ONLY**. Retained for legacy script compatibility; use `table4_counterfactual_prerequisite_ranking.csv` and `table5_learning_path_comparison.csv`.
- **`reports/tables/table_literature_benchmark.csv`**: **SUPERSEDED**. Replaced by `reports/tables/table_literature_benchmark_FINAL.csv`.

---

### C. Supplemental System Audits & Diagnostic Reports (Informational)
These files document codebase inspections, UI/UX assessments, and diagnostic metadata:
- `reports/RESEARCH_AUDIT_REPORT.md`: Comprehensive audit report covering 12 evaluation axes and IEEE submission prerequisites.
- `reports/research_paper_results_showcase.md`: Narrative showcase describing key findings and defense Q&A.
- `reports/dataset_audit.md` & `dataset_audit.json`: Data integrity verification and split statistics.
- `reports/kc_graph_summary.md` & `kc_graph_audit.json`: Graph structure diagnostics.
- `reports/model_evaluation.md` & `model_evaluation.json`: Early model diagnostics.
- `reports/systematic_experiments.json`: Hyperparameter grid sweep records.
- `reports/ui_ux_system_audit.md`: UI/UX design token evaluation.
- `reports/issue_log.md` & `limitations.md`: Known experimental limitations and architectural boundary definitions.

---

## 2. Key Manuscript-Ready Statistics Quick Reference

| Metric / Finding | Exact Canonical Value | Verification Script / Evidence File |
| :--- | :---: | :--- |
| **GNN Test ROC-AUC ($\text{Seed } 42$)** | **0.7060** | `evidence_final_20261111/metrics_by_model.csv` |
| **GNN Test PR-AUC** | **0.8775** | `evidence_final_20261111/metrics_by_model.csv` |
| **GNN Multi-Seed ROC-AUC** | **0.7073 $\pm$ 0.0104** | `reports/tables/table7_multi_seed_robustness.csv` |
| **GNN Multi-Seed Log Loss** | **0.4570 $\pm$ 0.0014** | `reports/tables/table7_multi_seed_robustness.csv` |
| **Validation-Selected Threshold ($\tau^*$)** | **0.8256** | `reports/tables/table3_test_performance_comparison.csv` |
| **Balanced Accuracy at $\tau^*$** | **0.6475** | `reports/tables/table3_test_performance_comparison.csv` |
| **Multi-Seed Balanced Accuracy** | **0.6575 $\pm$ 0.0217** | `reports/tables/table_literature_benchmark_FINAL.csv` |
| **Structural Path Divergence Rate** | **45.0%** (45 / 100 pairs) | `scripts/run_sensitivity_and_divergence.py` |
| **Sensitivity Scaling ($\theta \in [0.60 \dots 0.95]$)** | **Monotonic, Rank-Invariant** | `reports/tables/table_sensitivity_analysis.csv` |
| **Automated Tests Passing** | **11 / 11 (100%)** | `reports/TEST_RUN_FINAL.txt` |

