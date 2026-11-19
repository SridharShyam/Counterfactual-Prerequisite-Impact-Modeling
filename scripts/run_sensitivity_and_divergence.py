"""
Script to compute Delta-Sensitivity Analysis and Multi-Student Path Divergence
using the canonical GNN configuration (embedding_dim=64, hidden_dim=64, epochs=12, seed=42).
Uses the exact sequential training RNG stream (Baseline -> MLP -> GNN) established in
run_reproducible_pipeline.py to guarantee 100% numerical identity with canonical artifacts.

Outputs:
1. reports/tables/table_sensitivity_analysis.csv
2. reports/evidence_final_20261111/tables/table_sensitivity_analysis.csv
3. reports/figures/counterfactual_sensitivity.png & .pdf
4. reports/figures_final/counterfactual_sensitivity.png & .pdf
5. reports/evidence_final_20261111/figures/counterfactual_sensitivity.png & .pdf
"""
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

proj_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cflp.config import load_config, get_project_root
from cflp.data.loader import DatasetLoader
from cflp.graph.builder import KCGraphBuilder
from cflp.models.predictors import KnowledgeStateEstimator, BaselineLogisticPredictor, GraphNeuralPredictor
from cflp.counterfactual.impact import CounterfactualImpactEngine
from cflp.paths.generator import LearningPathGenerator
from scripts.train_models import StudentPredictionDataset, build_normalized_adjacency

def main():
    print("=" * 70)
    print("C-PrIM CANONICAL SENSITIVITY & DIVERGENCE EVALUATION (SEQUENTIAL RNG)")
    print("=" * 70)

    # Set canonical random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = get_project_root()
    cfg = load_config()
    ds_root = Path(cfg["data"]["dataset_root_abs"])

    reports_dir = root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    figures_final_dir = reports_dir / "figures_final"
    evidence_dir = reports_dir / "evidence_final_20261111"
    evidence_tables_dir = evidence_dir / "tables"
    evidence_figures_dir = evidence_dir / "figures"

    for d in [tables_dir, figures_dir, figures_final_dir, evidence_tables_dir, evidence_figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading dataset and building pedagogical graph...")
    loader = DatasetLoader(str(ds_root))
    estimator = KnowledgeStateEstimator(num_concepts=1175)
    builder = KCGraphBuilder(str(ds_root / "metadata"))
    graph = builder.build_graph()
    adj_matrix = build_normalized_adjacency(graph, num_nodes=1175)

    train_records = loader.load_interaction_sequences(split="train", val_fold=4, max_rows=3000)
    val_records = loader.load_interaction_sequences(split="val", val_fold=4, max_rows=1000)
    test_records = loader.load_interaction_sequences(split="test", max_rows=1500)

    train_ds = StudentPredictionDataset(train_records, estimator, graph)
    val_ds = StudentPredictionDataset(val_records, estimator, graph)
    test_ds = StudentPredictionDataset(test_records, estimator, graph)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("\n[2/4] Training Canonical Pipeline (Baseline -> MLP -> GNN sequentially under Seed 42)...")
    
    # 1. Baseline Logistic
    baseline_model = BaselineLogisticPredictor(num_concepts=1175)
    b_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(12):
        baseline_model.train()
        for b in train_loader:
            b_opt.zero_grad()
            p = baseline_model(b["target_kc"], b["target_mastery"], b["prereq_mastery"], b["history_len"])
            l = criterion(p, b["label"])
            l.backward()
            b_opt.step()
        baseline_model.eval()
        with torch.no_grad():
            for b in val_loader:
                _ = baseline_model(b["target_kc"], b["target_mastery"], b["prereq_mastery"], b["history_len"])

    # 2. No-Graph MLP
    class NoGraphMLP(nn.Module):
        def __init__(self, num_concepts=1175, embedding_dim=64, hidden_dim=64):
            super().__init__()
            self.emb = nn.Embedding(num_concepts, embedding_dim)
            self.net = nn.Sequential(
                nn.Linear(1 + embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, m_vec, target_kc):
            batch_size = m_vec.size(0)
            target_m = m_vec[torch.arange(batch_size), target_kc].unsqueeze(-1)
            target_e = self.emb(target_kc)
            x = torch.cat([target_m, target_e], dim=-1)
            return torch.sigmoid(self.net(x).squeeze(-1))

    mlp_model = NoGraphMLP(num_concepts=1175, embedding_dim=64, hidden_dim=64)
    m_opt = torch.optim.Adam(mlp_model.parameters(), lr=0.003)

    for epoch in range(12):
        mlp_model.train()
        for b in train_loader:
            m_opt.zero_grad()
            p = mlp_model(b["mastery_vec"], b["target_kc"])
            l = criterion(p, b["label"])
            l.backward()
            m_opt.step()
        mlp_model.eval()
        with torch.no_grad():
            for b in val_loader:
                _ = mlp_model(b["mastery_vec"], b["target_kc"])

    # 3. Canonical GNN Predictor (Dim 64, 12 Epochs)
    gnn_model = GraphNeuralPredictor(num_concepts=1175, embedding_dim=64, hidden_dim=64, dropout=0.20)
    g_opt = torch.optim.Adam(gnn_model.parameters(), lr=0.003, weight_decay=1e-4)

    for epoch in range(12):
        gnn_model.train()
        total_loss = 0.0
        for batch in train_loader:
            g_opt.zero_grad()
            preds = gnn_model(batch["mastery_vec"], batch["target_kc"], adj_matrix)
            loss = criterion(preds, batch["label"])
            loss.backward()
            g_opt.step()
            total_loss += loss.item() * len(batch["label"])
        gnn_model.eval()
        with torch.no_grad():
            for b in val_loader:
                _ = gnn_model(b["mastery_vec"], b["target_kc"], adj_matrix)
        if (epoch + 1) % 4 == 0 or epoch == 11:
            print(f"  GNN Epoch {epoch+1}/12 - Train Loss: {total_loss / len(train_ds):.4f}")

    # Initialize engines
    gnn_model.eval()
    cf_engine = CounterfactualImpactEngine(gnn_model, graph, adj_matrix)
    path_gen = LearningPathGenerator(graph)

    # -------------------------------------------------------------------------
    # Task 2: Delta-Sensitivity Analysis
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Delta-Sensitivity Analysis across theta in [0.60, 0.75, 0.85, 0.95]...")
    sample_student = test_records[0]
    sm = estimator.estimate_mastery(sample_student["concepts"], sample_student["responses"])
    student_uid = sample_student["uid"]
    print(f"  Representative Student: UID {student_uid}")

    intervention_levels = [0.60, 0.75, 0.85, 0.95]
    sensitivity_rows = []

    for target_kc in [4, 5]:
        target_name = loader.get_kc_name(target_kc, "en")
        print(f"\n  Evaluating Target KC {target_kc}: '{target_name}'")

        level_impacts = {}
        candidate_meta = {}
        for theta in intervention_levels:
            rankings = cf_engine.compute_counterfactual_impacts(
                target_kc=target_kc,
                student_mastery=sm,
                intervention_level=theta,
                enforce_improvement=True
            )
            level_impacts[theta] = {item["kc_id"]: item["estimated_impact"] for item in rankings}
            for item in rankings:
                if item["kc_id"] not in candidate_meta:
                    candidate_meta[item["kc_id"]] = {
                        "name": loader.get_kc_name(item["kc_id"], "en"),
                        "current_mastery": item["current_mastery"]
                    }

        # Sort candidate KCs by default theta=0.85 impact descending
        sorted_candidates = sorted(candidate_meta.keys(), key=lambda c: level_impacts[0.85].get(c, 0.0), reverse=True)

        for rank_idx, cid in enumerate(sorted_candidates, 1):
            imp_60 = level_impacts[0.60].get(cid, 0.0)
            imp_75 = level_impacts[0.75].get(cid, 0.0)
            imp_85 = level_impacts[0.85].get(cid, 0.0)
            imp_95 = level_impacts[0.95].get(cid, 0.0)

            is_monotonic = (imp_60 <= imp_75 <= imp_85 <= imp_95)
            stability = f"Rank {rank_idx} (Monotonic)" if is_monotonic else f"Rank {rank_idx}"

            row = {
                "Student_UID": student_uid,
                "Target_KC": target_kc,
                "Target_Name": target_name,
                "Candidate_KC": cid,
                "Candidate_Name": candidate_meta[cid]["name"],
                "Current_Mastery": round(candidate_meta[cid]["current_mastery"], 4),
                "Impact_theta_0.60": round(imp_60, 4),
                "Impact_theta_0.75": round(imp_75, 4),
                "Impact_theta_0.85_default": round(imp_85, 4),
                "Impact_theta_0.95": round(imp_95, 4),
                "Ranking_Stability": stability
            }
            sensitivity_rows.append(row)
            print(f"    Candidate KC {cid} ('{candidate_meta[cid]['name']}'): "
                  f"theta=0.60: +{imp_60*100:.2f}%, theta=0.75: +{imp_75*100:.2f}%, "
                  f"theta=0.85: +{imp_85*100:.2f}%, theta=0.95: +{imp_95*100:.2f}% [{stability}]")

    df_sensitivity = pd.DataFrame(sensitivity_rows)
    
    sens_out_path = tables_dir / "table_sensitivity_analysis.csv"
    sens_evid_path = evidence_tables_dir / "table_sensitivity_analysis.csv"
    df_sensitivity.to_csv(sens_out_path, index=False)
    df_sensitivity.to_csv(sens_evid_path, index=False)
    print(f"\n  Saved sensitivity analysis table to:\n    - {sens_out_path}\n    - {sens_evid_path}")

    # Generate publication-grade sensitivity figure
    plt.figure(figsize=(8.5, 5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cbd5e1"
    plt.rcParams["axes.linewidth"] = 0.8

    # Plot Target KC 5 (Authoritative Target Concept matching Table II and Fig 9)
    # and Target KC 4 (Showcase)
    # We plot Target KC 5 as primary solid lines and Target KC 4 as dashed
    palette = ["#4338ca", "#06b6d4", "#f59e0b", "#10b981"]
    markers = ["o-", "s-", "^-", "d-"]

    kc5_rows = [r for r in sensitivity_rows if r["Target_KC"] == 5]
    for idx, r in enumerate(kc5_rows):
        y_vals = [r["Impact_theta_0.60"] * 100, r["Impact_theta_0.75"] * 100, 
                  r["Impact_theta_0.85_default"] * 100, r["Impact_theta_0.95"] * 100]
        lbl = f"KC {r['Candidate_KC']}: {r['Candidate_Name'][:24]}"
        plt.plot(intervention_levels, y_vals, markers[idx % len(markers)], 
                 color=palette[idx % len(palette)], label=lbl, linewidth=2.2, markersize=7)

    plt.axvline(x=0.85, color="#ef4444", linestyle=":", linewidth=1.5, label=r"Default $\theta = 0.85$")
    plt.title(r"Figure 11: Counterfactual Prerequisite Impact Sensitivity Across Intervention Levels ($\theta$)", fontsize=10, fontweight="bold", pad=12)
    plt.xlabel(r"Counterfactual Target Intervention Mastery ($\theta$)", fontsize=10, fontweight="bold")
    plt.ylabel("Estimated Target Probability Gain (%)", fontsize=10, fontweight="bold")
    plt.xticks(intervention_levels, [f"{x:.2f}" for x in intervention_levels])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)
    plt.tight_layout()

    fig_targets = [
        figures_dir / "counterfactual_sensitivity",
        figures_final_dir / "counterfactual_sensitivity",
        figures_final_dir / "fig11_counterfactual_sensitivity",
        evidence_figures_dir / "counterfactual_sensitivity"
    ]
    for ft in fig_targets:
        plt.savefig(f"{ft}.png", dpi=300)
        plt.savefig(f"{ft}.pdf")
        print(f"  Saved: {ft}.png & .pdf")
    plt.close()

    # -------------------------------------------------------------------------
    # Task 3: Structural Path Divergence Verification
    # -------------------------------------------------------------------------
    print("\n[4/4] Verifying Multi-Student Learning Path Divergence across test student-target pairs...")
    candidate_targets = [1, 2, 4, 5, 8, 9, 10, 12, 14, 15]
    eligible_pairs = 0
    divergent_paths = 0

    for s_idx, record in enumerate(test_records[:50]):
        student_mastery = estimator.estimate_mastery(record["concepts"], record["responses"])
        for tid in candidate_targets:
            ancs = list(nx.ancestors(graph, tid))
            if len(ancs) >= 2:
                eligible_pairs += 1
                try:
                    rk = cf_engine.compute_counterfactual_impacts(target_kc=tid, student_mastery=student_mastery, intervention_level=0.85)
                    p_res = path_gen.generate_path(target_kc=tid, student_mastery=student_mastery, impact_rankings=rk, max_path_length=5)
                    p_kcs = [st["kc_id"] for st in p_res["steps"]]
                    w_kcs = sorted(ancs, key=lambda c: student_mastery[c])[:len(ancs)] + [tid]
                    is_divergent = (p_kcs != w_kcs)
                    if is_divergent:
                        divergent_paths += 1
                except Exception as e:
                    print(f"    Error on student {record['uid']}, target {tid}: {e}")

    divergence_pct = round((divergent_paths / eligible_pairs * 100) if eligible_pairs > 0 else 0.0, 2)
    print(f"\n  DIVERGENCE AUDIT RESULT:")
    print(f"    Total Evaluated Pairs: {eligible_pairs}")
    print(f"    Divergent Path Count:  {divergent_paths}")
    print(f"    Divergence Percentage: {divergence_pct}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
