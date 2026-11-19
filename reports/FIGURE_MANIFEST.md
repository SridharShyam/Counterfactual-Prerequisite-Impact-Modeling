# C-PrIM Publication Figure Manifest (Canonical Evidence Run)

> [!NOTE]
> **Audit Note (Corrected on 2026-11-12):** Corrected GNN test PR-AUC in Figure 6 caption from $0.8872$ to $0.8775$ to match `reports/evidence_final_20261111/metrics_by_model.csv` exactly.

This manifest indexes the publication-ready figures generated from the canonical experimental run (`reports/evidence_final_20261111/`) for the IEEE manuscript on **Counterfactual Learning Paths (C-PrIM)**.

All figures are compiled at **300 DPI** in both vector (**PDF**) and raster (**PNG**) formats, located in `reports/figures_final/`.

---

## 1. Summary of Available Figures

| # | Figure Filename (Base) | IEEE Paper Section | Recommended Caption | Data Source / Basis |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `fig1_cflp_pipeline_architecture` | Section IV.A (System Design), Fig. 1 | End-to-end architecture of the C-PrIM framework, illustrating interaction sequence processing, knowledge state estimation, curriculum graph message passing, counterfactual impact estimation, and topological learning path generation. | Structural Pipeline Design |
| **2** | `fig2_knowledge_graph_overview` | Section III.B (Dataset & Preprocessing), Fig. 2 | Structural topology of the XES3G5M knowledge component (KC) curriculum graph, showing the in-degree distribution of prerequisite relationships across 1,175 catalogued concepts and 1,062 directed edges. | Pedagogical Graph Inventory |
| **3** | `fig3_representative_prerequisite_subgraph` | Section IV.B (Knowledge Graph Formalization), Fig. 3 | Representative prerequisite dependency subgraph showing directed curricular relationships among elementary arithmetic and combinatorial reasoning concepts (KCs 0, 2, 4, 5, 8, 9). | Subgraph Extraction |
| **4** | `fig4_training_validation_curves` | Section V.B (Model Convergence & Training Dynamics), Fig. 4 | Training loss convergence (Binary Cross-Entropy) and validation ROC-AUC trajectories across 12 training epochs comparing the Proposed GNN, Ablation No-Graph MLP, and Baseline Logistic models. | Canonical Training Logs (Seed 42) |
| **5** | `fig5_roc_curves` | Section V.C (Predictive Performance Comparison), Fig. 5 | Receiver Operating Characteristic (ROC) curves on the held-out test partition ($N=1,500$), demonstrating the discriminative superiority of the Proposed GNN ($\text{AUC} = 0.7060$) over No-Graph MLP ($\text{AUC} = 0.6995$) and Baseline Logistic ($\text{AUC} = 0.7035$). | Test Predictions (`test_predictions.npz`) |
| **6** | `fig6_precision_recall_curves` | Section V.C (Predictive Performance Comparison), Fig. 6 | Precision-Recall (PR) curves on the imbalanced held-out test set ($78.47\%$ positive response prevalence), highlighting the precision advantage of the Proposed GNN ($\text{PR-AUC} = 0.8775$) across operational recall regimes. | Test Predictions (`test_predictions.npz`) |
| **7** | `fig7_metric_comparison_plot` | Section V.C (Predictive Performance Comparison), Fig. 7 | Comparative evaluation across ROC-AUC, PR-AUC, Log Loss, and Brier Score, showing calibrated probability estimation and error reduction achieved via relational graph convolutions. | Test Metrics (`metrics_by_model.csv`) |
| **8** | `fig8_multiseed_robustness` | Section V.E (Robustness & Multi-Seed Validation), Fig. 8 | Test ROC-AUC stability across three independent random seeds ($\{42, 123, 999\}$), demonstrating tight empirical consistency ($\text{Mean} = 0.7073 \pm 0.0104$) without performance collapse. | Multi-Seed Evaluation (`table7_multi_seed_robustness.csv`) |
| **9** | `fig9_counterfactual_impact_distribution` | Section VI.A (Counterfactual Prerequisite Ranking), Fig. 9 | Candidate prerequisite counterfactual impact scores ($\delta$) for target KC 5 (*Route matching problem*) under default intervention ($\theta = 0.85$), identifying KC 4 and KC 0 as highest-leverage instructional interventions. | Counterfactual Analysis Engine |
| **10** | `fig10_personalized_path_examples` | Section VI.B (Personalized Path Generation), Fig. 10 | Comparison of pedagogical learning sequences generated for Student 8572 targeting KC 5: the Proposed C-PrIM path strictly enforces prerequisite transitive closure ($\text{Steps}: 0 \to 2 \to 4 \to 5$), whereas the heuristic Weakest-First baseline induces pedagogical ordering violations ($\text{Steps}: 0 \to 4 \to 2 \to 5$). | Path Generator (`table5_learning_path_comparison.csv`) |
| **11** | `counterfactual_sensitivity` | Section VI.C (Sensitivity Analysis), Fig. 11 | Sensitivity of counterfactual target impact estimates across intervention levels $\theta \in [0.60, 0.75, 0.85, 0.95]$, exhibiting monotonic response scaling and ranking order invariance. | Delta-Sensitivity Analysis (`table_sensitivity_analysis.csv`) |

---

## 2. Selection Guide for 3 to 5 Manuscript Figures

For an IEEE conference or transactions paper with 3 to 5 figure slots, the following subsets are recommended:

### Option A: Standard 4-Figure Selection (System + Performance + Mechanism + Case Study)
1. **Fig. 1 (`fig1_cflp_pipeline_architecture.pdf`)** $\to$ *Section IV.A*: Demonstrates overall system design and pipeline stages.
2. **Fig. 5 (`fig5_roc_curves.pdf`)** $\to$ *Section V.C*: Core predictive evaluation showing ROC curves on test set.
3. **Fig. 10 (`fig10_personalized_path_examples.pdf`)** $\to$ *Section VI.B*: Key educational takeaway proving topological ordering vs. heuristic failure.
4. **Fig. 11 (`counterfactual_sensitivity.pdf`)** $\to$ *Section VI.C*: Proves numerical stability of the counterfactual mechanism across $\theta$.

### Option B: Deep Empirical 5-Figure Selection (Includes Calibration & Robustness)
1. **Fig. 1 (`fig1_cflp_pipeline_architecture.pdf`)** $\to$ *Section IV.A*: Framework architecture.
2. **Fig. 5 (`fig5_roc_curves.pdf`)** $\to$ *Section V.C*: ROC discrimination curves.
3. **Fig. 8 (`fig8_multiseed_robustness.pdf`)** $\to$ *Section V.E*: Multi-seed reproducibility and stability.
4. **Fig. 10 (`fig10_personalized_path_examples.pdf`)** $\to$ *Section VI.B*: Prerequisite-preserving path sequence comparison.
5. **Fig. 11 (`counterfactual_sensitivity.pdf`)** $\to$ *Section VI.C*: Intervention intensity sensitivity curve.

