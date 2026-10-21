# Counterfactual Prerequisite Impact Modeling & Learning Path Report

## 1. Research Context & Formulation
- **Student ID:** `8572` (Real student sequence from disjoint holdout test partition).
- **Target Concept ($T$):** KC `4` — *"Object quantity combination"*.
- **Factual Estimated Target Performance ($\hat{M}_T^{F}$):** **0.8080**
- **Impact Equation:**
  $$\hat{I}_i = \hat{M}_T^{CF(i)} - \hat{M}_T^{F}$$
  where $\hat{M}_T^{CF(i)}$ represents the model's predicted performance on target $T$ after setting prerequisite concept $i$'s mastery to a hypothetical intervention level $\theta=0.85$.

## 2. Model-Based Counterfactual Impact Ranking Table

| Rank | Candidate KC | Concept Name (English) | Current Mastery | Intervention Level | Factual Target | Intervened Target | Estimated Impact ($I_i$) | Relationship |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `0` | Other types in the multiplication principle | 0.5000 | 0.85 | 0.8080 | 0.8129 | **+0.0049** | direct_prerequisite |
| 2 | `2` | Item matching | 0.6280 | 0.85 | 0.8080 | 0.8111 | **+0.0030** | direct_prerequisite |

## 3. Generated Personalized Learning Path

| 1 | `0` | **Other types in the multiplication principle** | 50.00% | +0.49% | Prerequisite Concept: Current mastery is 50.0%. Improves target probability by +0.5%. |
| 2 | `2` | **Item matching** | 62.80% | +0.30% | Prerequisite Concept: Current mastery is 62.8%. Improves target probability by +0.3%. |
| 3 | `4` | **Object quantity combination** | 50.00% | +0.00% | Target Concept: Goal of the personalized learning progression. |

## 4. Sensitivity Analysis Across Intervention Levels
- Tested $\theta \in [0.60, 0.75, 0.85, 0.95]$.
- Impact scales monotonically with the intervention intensity, preserving the ranking stability of primary foundational prerequisites.

## 5. Epistemological and Scientific Caveats
- These predictions represent **model-based counterfactual evaluations** derived from message-passing graph neural networks.
- They do not constitute guaranteed causal improvements in real-world human learners without controlled randomized educational trials.

*Generated deterministically by `scripts/run_experiment.py`.*
