# Knowledge Component (KC) Graph Audit & Summary Report

## 1. Graph Semantics & Methodological Justification
- **Graph Type:** Hierarchy-Derived Candidate Prerequisite Graph (Taxonomy Cluster Projection)
- **Scientific Caveat:** Graph represents educational taxonomy and co-occurrence constraints, not experimentally verified causal links.
- **Construction Process:**
  1. All 1,175 officially defined knowledge components are mapped as primary nodes with dual English and Chinese route metadata.
  2. Question taxonomy hierarchies (`Module -> Topic -> Subtopic -> Leaf Concept`) are extracted from 7,652 annotated problems.
  3. Co-membership within specific subtopic clusters defines candidate prerequisite progression edges.
  4. Directional acyclicity is strictly enforced and verified using topological ordering to support deterministic downstream learning-path generation.

## 2. Structural Graph Statistics
- **Total KC Nodes:** 1,175
- **Total Candidate Edges:** 1,062
- **Directed Acyclic Graph (DAG) Verified:** **True**
- **Isolated Nodes (KCs without immediate cluster siblings):** 443
- **Weakly Connected Components:** 638
- **Average In-Degree:** 0.904 (Max: 6)
- **Average Out-Degree:** 0.904 (Max: 6)

## 3. Storage and Reproducibility
- Serialized Graph: `artifacts/kc_relationship_graph.json`
- Audit Metrics: `reports/kc_graph_audit.json`
