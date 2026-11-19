# C-PrIM: Numerical Results Ready for Paper

> [!NOTE]
> **Audit Note (Corrected on 2026-11-12):** Corrected validation-selected threshold $\tau^*$ from $0.7786$ to $0.8256$ and GNN test PR-AUC from $0.8872$ to $0.8775$ (with Baseline Logistic PR-AUC at $0.8655$ and No-Graph MLP at $0.8641$, updating the Absolute Delta to $+0.0120$) to match `reports/evidence_final_20261111/metrics_by_model.csv` exactly.

This document compiles all definitive numerical results from the canonical experimental evidence package (`reports/evidence_final_20261111/`, random seeds $\{42, 123, 999\}$, embedding/hidden dim $64$, $12$ epochs).

---

## (a) Table I Equivalent — Predictive Performance on Held-Out Test Set ($N = 1,500$)
**Source:** [table3_test_performance_comparison.csv](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/tables/table3_test_performance_comparison.csv) and [metrics_by_model.csv](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/evidence_final_20261111/metrics_by_model.csv)

| Metric | Baseline Logistic ($\tau = 0.50$) | Ablation No-Graph MLP ($\tau = 0.50$) | Proposed GNN ($\tau = 0.50$) | Proposed GNN ($\tau^* = 0.8256$) | Absolute Delta ($\text{GNN}_{\tau^*} \text{ vs. Baseline}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.7035 | 0.6995 | **0.7060** | **0.7060** | +0.0025 |
| **PR-AUC** | 0.8655 | 0.8641 | **0.8775** | **0.8775** | +0.0120 |
| **Accuracy** | 0.8040 | 0.8020 | 0.8033 | 0.6160 | -0.1880 |
| **Balanced Accuracy** | 0.5516 | 0.5488 | 0.5522 | **0.6475** | +0.0959 |
| **Precision** | 0.8047 | 0.8040 | 0.8029 | **0.8789** | +0.0742 |
| **Recall (Sensitivity)** | 0.9907 | 0.9881 | 0.9932 | 0.5922 | -0.3985 |
| **F1 Score** | 0.8880 | 0.8864 | 0.8880 | 0.7076 | -0.1804 |
| **Log Loss (BCE)** | 0.4665 | 0.4746 | **0.4562** | **0.4562** | -0.0103 |
| **Brier Score** | 0.1465 | 0.1457 | **0.1446** | **0.1446** | -0.0019 |

*Note on Class Imbalance:* Held-out test set exhibits $78.47\%$ positive response prevalence ($1,177$ correct vs. $323$ incorrect). Freezing validation-tuned threshold $\tau^* = 0.8256$ lifts balanced accuracy from $55.22\%$ to $64.75\%$ and precision to $87.89\%$.

---

## (b) Table II Equivalent — Ablation Study
**Source:** [table6_ablation_study.csv](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/tables/table6_ablation_study.csv)

| Model Variant | Test ROC-AUC | Log Loss | Brier Score | Prerequisite Topology Preserved |
| :--- | :---: | :---: | :---: | :---: |
| **Full Proposed Framework (GNN + Prerequisite Graph)** | **0.7060** | **0.4562** | **0.1446** | **Yes (Full DAG Message Passing)** |
| **Ablation A: No Prerequisite Graph (MLP Target Only)** | 0.6995 | 0.4746 | 0.1457 | No (Isolated Target Embedding) |
| **Ablation B: Baseline Logistic Model (Handcrafted Features)** | 0.7035 | 0.4665 | 0.1465 | Partial (Scalar Prereq Mastery Average) |

---

## (c) Table III Equivalent — Multi-Seed Robustness Evaluation
**Source:** [table7_multi_seed_robustness.csv](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/tables/table7_multi_seed_robustness.csv)

| Run / Seed | Test ROC-AUC | Log Loss | Test Balanced Accuracy ($\tau^*$) |
| :--- | :---: | :---: | :---: |
| **Seed 42** | 0.7060 | 0.4562 | 0.6475 |
| **Seed 123** | 0.7183 | 0.4562 | 0.6826 |
| **Seed 999** | 0.6977 | 0.4587 | 0.6425 |
| **Mean $\pm$ Std** | **0.7073 $\pm$ 0.0104** | **0.4570 $\pm$ 0.0014** | **0.6575 $\pm$ 0.0217** |

---

## (d) Sensitivity Analysis Across Intervention Intensities
**Source:** [table_sensitivity_analysis.csv](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/tables/table_sensitivity_analysis.csv)  
**Evaluated On:** Student UID `8572` (Test Sequence 0) via `scripts/run_sensitivity_and_divergence.py`

| Target KC | Candidate Prerequisite | Current Mastery | $\theta = 0.60$ | $\theta = 0.75$ | $\theta = 0.85$ (Default) | $\theta = 0.95$ | Ranking Stability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KC 4** (*Object quantity combination*) | **KC 0** (*Multiplication principle*) | 0.5000 | +0.40% | +1.07% | **+1.51%** | +1.99% | **Rank 1 (Monotonic)** |
| **KC 4** (*Object quantity combination*) | **KC 2** (*Item matching*) | 0.6280 | +0.28% | +0.35% | **+0.65%** | +0.97% | **Rank 2 (Monotonic)** |
| **KC 5** (*Route matching problem*) | **KC 0** (*Multiplication principle*) | 0.5000 | +0.40% | +1.07% | **+1.51%** | +1.99% | **Rank 1 (Monotonic)** |
| **KC 5** (*Route matching problem*) | **KC 4** (*Object quantity combination*) | 0.5000 | +0.39% | +0.98% | **+1.36%** | +1.72% | **Rank 2 (Monotonic)** |
| **KC 5** (*Route matching problem*) | **KC 2** (*Item matching*) | 0.6280 | +0.28% | +0.35% | **+0.65%** | +0.97% | **Rank 3 (Monotonic)** |

*Note on Stability and Monotonicity:* Across all candidate concepts and target nodes, estimated downstream counterfactual impact scales strictly monotonically with respect to intervention intensity $\theta$. The relative candidate priority ranking remains perfectly stable and invariant across all tested intensity regimes ($\theta \in [0.60, 0.95]$), with KC 0 providing the highest marginal impact followed by KC 4 and KC 2.

---

## (e) Literature Benchmark Comparison Table
**Source:** [table_literature_benchmark_FINAL.csv](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/tables/table_literature_benchmark_FINAL.csv)

| Model | Year | Venue | Architecture | Input Features | Reported AUC | Reported Accuracy | Source |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **DKT** | 2015 | NeurIPS 2015 | Recurrent Neural Network (LSTM / RNN) | Sequential concept-response tuples (seq len $\le 200$) | 0.785 - 0.835 | 0.742 - 0.760 | Piech et al. (2015) / XES3G5M Benchmark (Liu et al., 2023) |
| **SAKT** | 2019 | EDM 2019 | Self-Attention Transformer Encoder | Interaction sequence self-attention | 0.769 - 0.795 | 0.730 - 0.755 | Pandey & Karypis (2019) / XES3G5M Benchmark (Liu et al., 2023) |
| **GKT** | 2019 | WI 2019 | Graph Neural Network (GNN) on Transition Graph | Transition statistics + response sequence message passing | 0.772 - 0.799 | 0.735 - 0.752 | Nakagawa et al. (2019) / XES3G5M Benchmark (Liu et al., 2023) |
| **AKT** | 2020 | ACM KDD 2020 | Context-Aware Attention with Psychometric Decay | Question IDs, Concept IDs, responses, relative distance | 0.820 - 0.843 | 0.765 - 0.782 | Ghosh et al. (2020) / XES3G5M Benchmark (Liu et al., 2023) |
| **C-PrIM Proposed GNN** | 2026 | IEEE Trans./Conf. (This Study) | GCN Relational Graph Neural Network | Recency-weighted mastery vector + DAG adjacency (zero text/ID leakage) | **0.7073 $\pm$ 0.0104** | 0.6160 (BalAcc 0.6475; Multi-Seed BalAcc 0.6575 $\pm$ 0.0217) | Local Empirical Evaluation (Canonical Run: `reports/evidence_final_20261111`) |

---

## (f) Structural Path Divergence Statistic
**Source:** [IEEE_COMPARATIVE_EVALUATION.md](file:///c:/Users/shyam/Downloads/Counterfactual%20Learning%20Path/CFLP_Project/reports/IEEE_COMPARATIVE_EVALUATION.md#L86-L92) and verified by `scripts/run_sensitivity_and_divergence.py`

- **Evaluated Pairs:** $100$ test student-target pairs ($50$ test students across candidate targets with $\ge 2$ prerequisite ancestors).
- **Divergent Cases:** $45$ pairs.
- **Structural Path Divergence Rate:** **$45.0\%$**
- **Pedagogical Finding:** In all $45$ divergent instances, the heuristic baseline (**Weakest-Mastery-First**) selected prerequisite concepts that violated DAG topological ordering (e.g., prescribing KC 4 before KC 2, or KC 2 before KC 0), while C-PrIM preserved strict pedagogical dependency closure.

