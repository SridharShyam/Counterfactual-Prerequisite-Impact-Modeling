"""
Script to build, validate, and audit the Knowledge Component Relationship Graph.
Generates:
- artifacts/kc_relationship_graph.json
- reports/kc_graph_audit.json
- reports/kc_graph_summary.md
"""
import os
import json
from pathlib import Path
from cflp.config import load_config, get_project_root
from cflp.graph.builder import KCGraphBuilder

def run_build_graph():
    print("=" * 65)
    print("BUILDING AND AUDITING KC RELATIONSHIP GRAPH")
    print("=" * 65)

    root = get_project_root()
    cfg = load_config()
    ds_root = Path(cfg["data"]["dataset_root_abs"])
    meta_dir = ds_root / "metadata"

    artifacts_dir = root / "artifacts"
    reports_dir = root / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    builder = KCGraphBuilder(str(meta_dir))
    graph = builder.build_graph()
    audit_data = builder.audit_graph()

    print(f"Graph Construction Complete:")
    print(f"- Total Nodes (KCs): {audit_data['num_nodes']}")
    print(f"- Total Directed Edges: {audit_data['num_edges']}")
    print(f"- Is Directed Acyclic Graph (DAG): {audit_data['is_dag']}")
    print(f"- Isolated Nodes: {audit_data['num_isolated_nodes']}")
    print(f"- Weakly Connected Components: {audit_data['num_weakly_connected_components']}")
    print(f"- Average In-Degree: {audit_data['avg_in_degree']:.2f}")

    # Export graph artifacts
    graph_json_path = artifacts_dir / "kc_relationship_graph.json"
    builder.export_graph_json(str(graph_json_path))

    # Export audit reports
    audit_json_path = reports_dir / "kc_graph_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    summary_md_path = reports_dir / "kc_graph_summary.md"
    md_content = f"""# Knowledge Component (KC) Graph Audit & Summary Report

## 1. Graph Semantics & Methodological Justification
- **Graph Type:** {audit_data['semantics_label']}
- **Scientific Caveat:** {audit_data['causal_disclaimer']}
- **Construction Process:**
  1. All 1,175 officially defined knowledge components are mapped as primary nodes with dual English and Chinese route metadata.
  2. Question taxonomy hierarchies (`Module -> Topic -> Subtopic -> Leaf Concept`) are extracted from 7,652 annotated problems.
  3. Co-membership within specific subtopic clusters defines candidate prerequisite progression edges.
  4. Directional acyclicity is strictly enforced and verified using topological ordering to support deterministic downstream learning-path generation.

## 2. Structural Graph Statistics
- **Total KC Nodes:** {audit_data['num_nodes']:,}
- **Total Candidate Edges:** {audit_data['num_edges']:,}
- **Directed Acyclic Graph (DAG) Verified:** **{audit_data['is_dag']}**
- **Isolated Nodes (KCs without immediate cluster siblings):** {audit_data['num_isolated_nodes']}
- **Weakly Connected Components:** {audit_data['num_weakly_connected_components']}
- **Average In-Degree:** {audit_data['avg_in_degree']:.3f} (Max: {audit_data['max_in_degree']})
- **Average Out-Degree:** {audit_data['avg_out_degree']:.3f} (Max: {audit_data['max_out_degree']})

## 3. Storage and Reproducibility
- Serialized Graph: `artifacts/kc_relationship_graph.json`
- Audit Metrics: `reports/kc_graph_audit.json`
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Graph builder completed successfully.")
    print(f"- Exported to: {graph_json_path}")
    print(f"- Summary report: {summary_md_path}")

if __name__ == "__main__":
    run_build_graph()
