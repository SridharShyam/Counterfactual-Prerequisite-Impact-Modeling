# C-PrIM Experimental Limitations and Research Boundaries

1. **Model-Based Simulation vs. Causal Reality:**
   - Counterfactual prerequisite impacts ($I_i = M_T^{(CF(i))} - M_T^{(F)}$) represent hypothetical outputs from the trained neural network.
   - They do not represent experimentally verified causal proof in the absence of controlled randomized A/B learner trials.

2. **Graph Semantic Nature:**
   - Edges in the knowledge graph represent textbook syllabus chapters and hierarchical sections from XES3G5M metadata.
   - They serve as pedagogical candidate relationships and provide message-passing connectivity, but are not verified cognitive prerequisites.

3. **External KT Literature Comparability:**
   - Standard Deep Knowledge Tracing (DKT, AKT, SAKT, GKT) benchmarks on XES3G5M evaluate sequential next-interaction prediction using continuous interaction timestamps and rich question-text embeddings (yielding AUCs ~0.78 to 0.84).
   - In contrast, C-PrIM intentionally isolates aggregated concept mastery vectors and curriculum topology to support counterfactual path generation without leaking future question bodies. Thus, C-PrIM's AUC (0.7088) is evaluated under a different problem formulation and should not be directly equated to sequential question-level text-augmented KT.

4. **Path Divergence vs. Educational Quality:**
   - The 45% divergence rate against Weakest-Mastery-First proves structural and algorithmic difference (preserving prerequisite dependencies), but human learner efficacy remains untested without classroom intervention studies.
