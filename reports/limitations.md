# Scientific and Methodological Limitations of the CFLP Prototype

## 1. Predictive Modeling vs. Causal Identification

- **Primary Scientific Distinction:** The Graph Neural Network (GNN) and baseline logistic models are **predictive estimators**, not structural causal models (SCMs).
- **Interpretation of Impact Scores:** The counterfactual impact estimate:
  $$\hat{I}_i = \hat{M}_T^{CF(i)} - \hat{M}_T^{F}$$
  measures the sensitivity of the model's prediction head to a mathematical perturbation in the input feature vector along graph message-passing channels. It **does not guarantee** that intervening pedagogically on a human learner in a real classroom will yield an equivalent $\hat{I}_i$ learning gain.
- **Unobserved Confounders:** Student learning trajectories are subject to real-world latent confounders (e.g., student fatigue, varying instructional methods, prior informal exposure) that are unobserved in the raw interaction logs.

## 2. Graph Semantics and Construction Limitations

- **Hierarchy-Derived vs. Verified Prerequisites:** The relationship graph is constructed from question taxonomy hierarchies (`Module -> Topic -> Subtopic -> Leaf Concept`) rather than human expert-annotated or randomized prerequisite ablation studies.
- **Transitive Assumptions:** Co-membership within a subtopic cluster implies topical synergy and curricular proximity, but cannot strictly establish asymmetric cognitive necessity (i.e., that mastery of concept $A$ is strictly mathematically required before understanding concept $B$).
- **Graph Pruning for Acyclicity:** Topological ordering was enforced to guarantee deterministic, cycle-free learning path generation. In real-world educational domains, bidirectional or spiral learning relationships can exist naturally.

## 3. Dataset Characteristics and Domain Generalization

- **Domain Specificity:** The XES3G5M dataset is exclusively mathematics-oriented, reflecting Chinese primary and secondary school mathematics curricula.
- **Translation Artifacts:** While leaf Knowledge Components were translated from Chinese into English using automated neural translation, nuances in localized pedagogical phrasing may not directly map to Western curricula (e.g., Common Core or UK National Curriculum).
- **Sparsity:** Of the 1,175 officially defined Knowledge Components in the metadata, approximately 865 appear actively in student interaction records. The remaining ~310 concepts have prior-only baseline states and limited empirical training signal.

## 4. Evaluation Boundaries

- **Disjoint Student Test Set:** Predictive metrics were evaluated on a strictly holdout test set with 0% student overlap, preventing data leakage.
- **Evaluation Target:** The evaluated target is binary response correctness on the next attempted question ($P(R_t=1)$), which serves as an empirical operationalization of knowledge mastery rather than a continuous latent trait score.
