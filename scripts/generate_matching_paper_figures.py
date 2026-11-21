import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve

proj = Path("C:/Users/shyam/Downloads/Counterfactual Learning Path/CFLP_Project")
paper_dir = proj / "paper_final_figures"
reports_fig_final = proj / "reports" / "figures_final"

# -------------------------------------------------------------------------
# Fig 1: Overall C-PrIM Pipeline Diagram
# File name: Fig1_Architecture.pdf / .png
# Title inside image: "Fig 1: Counterfactual Learning Path (C-PrIM) System Architecture & Pipeline"
# -------------------------------------------------------------------------
print("Generating Fig 1...")
fig1, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
ax.axis("off")
boxes = [
    {"text": "1. Student Interactions\n(XES3G5M Interaction Logs)", "x": 0.02, "w": 0.13, "color": "#e0e7ff", "border": "#4338ca"},
    {"text": "2. Knowledge State\nEstimator\n(Recency + Beta Prior)", "x": 0.18, "w": 0.13, "color": "#e0e7ff", "border": "#4338ca"},
    {"text": "3. Pedagogical Graph\n(Curriculum Taxonomy\nNormalized Adjacency)", "x": 0.34, "w": 0.14, "color": "#ede9fe", "border": "#6d28d9"},
    {"text": "4. GCN Predictor\n(Factual Success\nProbability)", "x": 0.51, "w": 0.13, "color": "#fef3c7", "border": "#b45309"},
    {"text": "5. Counterfactual\nEngine\n(Simulated Interventions)", "x": 0.67, "w": 0.14, "color": "#fee2e2", "border": "#b91c1c"},
    {"text": "6. Topological Path\nGenerator\n(Prerequisite Closure)", "x": 0.84, "w": 0.14, "color": "#dcfce7", "border": "#15803d"},
]
for b in boxes:
    rect = patches.FancyBboxPatch((b["x"], 0.25), b["w"], 0.5, boxstyle="round,pad=0.03", fc=b["color"], ec=b["border"], lw=2)
    ax.add_patch(rect)
    ax.text(b["x"] + b["w"]/2, 0.50, b["text"], ha="center", va="center", fontsize=9, fontweight="bold", color="#1e293b")

for i in range(len(boxes) - 1):
    x_start = boxes[i]["x"] + boxes[i]["w"] + 0.005
    x_end = boxes[i+1]["x"] - 0.005
    ax.annotate("", xy=(x_end, 0.50), xytext=(x_start, 0.50),
                arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))

ax.set_title("Fig 1: Counterfactual Learning Path (C-PrIM) System Architecture & Pipeline", fontsize=12, fontweight="bold", pad=12)
fig1.tight_layout()
fig1.savefig(paper_dir / "Fig1_Architecture.png", dpi=300)
fig1.savefig(paper_dir / "Fig1_Architecture.pdf")
plt.close(fig1)

# -------------------------------------------------------------------------
# Fig 2: ROC Curves
# File name: Fig2_ROC_Curves.pdf / .png
# Title inside image: "Fig 2: Receiver Operating Characteristic (ROC) on Held-Out Test Set"
# -------------------------------------------------------------------------
print("Generating Fig 2...")
preds = np.load(proj / "reports" / "evidence_final_20261111" / "predictions" / "test_predictions.npz")
y_test = preds["y_true"]
p_gnn = preds["gnn_probs"]
p_mlp = preds["mlp_probs"]
p_baseline = preds["baseline_probs"]

metrics_df = pd.read_csv(proj / "reports" / "evidence_final_20261111" / "metrics_by_model.csv")
gnn_auc = float(metrics_df[metrics_df["Model"].str.contains("Proposed GNN")]["ROC_AUC"].values[0])
base_auc = float(metrics_df[metrics_df["Model"].str.contains("Baseline")]["ROC_AUC"].values[0])
mlp_auc = float(metrics_df[metrics_df["Model"].str.contains("No-Graph MLP")]["ROC_AUC"].values[0])

fig2, ax2 = plt.subplots(figsize=(6.8, 5.2), dpi=300)
fpr_g, tpr_g, _ = roc_curve(y_test, p_gnn)
fpr_m, tpr_m, _ = roc_curve(y_test, p_mlp)
fpr_b, tpr_b, _ = roc_curve(y_test, p_baseline)

ax2.plot(fpr_g, tpr_g, label=f"Proposed GNN (AUC = {gnn_auc:.4f})", color="#4338ca", lw=2.2)
ax2.plot(fpr_b, tpr_b, label=f"Baseline Logistic (AUC = {base_auc:.4f})", color="#64748b", linestyle="--", lw=1.8)
ax2.plot(fpr_m, tpr_m, label=f"Ablation No-Graph MLP (AUC = {mlp_auc:.4f})", color="#0284c7", linestyle="-.", lw=1.8)
ax2.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance (AUC = 0.5000)")
ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
ax2.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
ax2.set_title("Fig 2: Receiver Operating Characteristic (ROC) on Held-Out Test Set", fontweight="bold", fontsize=10, pad=10)
ax2.legend(loc="lower right")
ax2.grid(True, linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig(paper_dir / "Fig2_ROC_Curves.png", dpi=300)
fig2.savefig(paper_dir / "Fig2_ROC_Curves.pdf")
plt.close(fig2)

# -------------------------------------------------------------------------
# Fig 3: Personalized Path Examples
# File name: Fig3_Path_Comparison.pdf / .png
# Title inside image: "Fig 3: Personalized Learning Path Sequence Comparison (Target KC 5)"
# -------------------------------------------------------------------------
print("Generating Fig 3...")
df_paths = pd.read_csv(proj / "reports" / "evidence_final_20261111" / "path_results.csv")
fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
prop_steps = df_paths[df_paths["Method"].str.contains("Proposed")]
weak_steps = df_paths[df_paths["Method"].str.contains("Weakest")]

ax3_1.barh([f"Step {r['Step_Number']}: KC {r['KC_ID']}" for _, r in prop_steps.iterrows()][::-1],
            [r["Current_Mastery"] * 100 for _, r in prop_steps.iterrows()][::-1], color="#4338ca", height=0.45)
ax3_1.set_xlabel("Initial Mastery (%)", fontweight="bold")
ax3_1.set_title("Proposed C-PrIM Path\n(Topological + Impact Ordering)", fontweight="bold")
ax3_1.set_xlim(0, 100)
ax3_1.grid(axis="x", linestyle="--", alpha=0.5)

ax3_2.barh([f"Step {r['Step_Number']}: KC {r['KC_ID']}" for _, r in weak_steps.iterrows()][::-1],
            [r["Current_Mastery"] * 100 for _, r in weak_steps.iterrows()][::-1], color="#64748b", height=0.45)
ax3_2.set_xlabel("Initial Mastery (%)", fontweight="bold")
ax3_2.set_title("Weakest-First Baseline\n(Heuristic Mastery Sort)", fontweight="bold")
ax3_2.set_xlim(0, 100)
ax3_2.grid(axis="x", linestyle="--", alpha=0.5)
fig3.suptitle("Fig 3: Personalized Learning Path Sequence Comparison (Target KC 5)", fontweight="bold", fontsize=11)
fig3.tight_layout()
fig3.savefig(paper_dir / "Fig3_Path_Comparison.png", dpi=300)
fig3.savefig(paper_dir / "Fig3_Path_Comparison.pdf")
plt.close(fig3)

# -------------------------------------------------------------------------
# Fig 4: Counterfactual Sensitivity
# File name: Fig4_Sensitivity.pdf / .png
# Title inside image: "Fig 4: Counterfactual Prerequisite Impact Sensitivity Across Intervention Levels ($\theta$)"
# -------------------------------------------------------------------------
print("Generating Fig 4...")
sens_df = pd.read_csv(proj / "reports" / "tables" / "table_sensitivity_analysis.csv")
intervention_levels = [0.60, 0.75, 0.85, 0.95]

fig4 = plt.figure(figsize=(8.5, 5), dpi=300)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8

palette = ["#4338ca", "#06b6d4", "#f59e0b", "#10b981"]
markers = ["o-", "s-", "^-", "d-"]

kc5_rows = sens_df[sens_df["Target_KC"] == 5]
for idx, (_, r) in enumerate(kc5_rows.iterrows()):
    y_vals = [r["Impact_theta_0.60"] * 100, r["Impact_theta_0.75"] * 100, 
              r["Impact_theta_0.85_default"] * 100, r["Impact_theta_0.95"] * 100]
    lbl = f"KC {r['Candidate_KC']}: {r['Candidate_Name'][:24]}"
    plt.plot(intervention_levels, y_vals, markers[idx % len(markers)], 
             color=palette[idx % len(palette)], label=lbl, linewidth=2.2, markersize=7)

plt.axvline(x=0.85, color="#ef4444", linestyle=":", linewidth=1.5, label=r"Default $\theta = 0.85$")
plt.title(r"Fig 4: Counterfactual Prerequisite Impact Sensitivity Across Intervention Levels ($\theta$)", fontsize=10, fontweight="bold", pad=12)
plt.xlabel(r"Counterfactual Target Intervention Mastery ($\theta$)", fontsize=10, fontweight="bold")
plt.ylabel("Estimated Target Probability Gain (%)", fontsize=10, fontweight="bold")
plt.xticks(intervention_levels, [f"{x:.2f}" for x in intervention_levels])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)
plt.tight_layout()

fig4.savefig(paper_dir / "Fig4_Sensitivity.png", dpi=300)
fig4.savefig(paper_dir / "Fig4_Sensitivity.pdf")
plt.close(fig4)

# -------------------------------------------------------------------------
# Fig 5: Multi-Seed Robustness Plot
# File name: Fig5_MultiSeed_Robustness.pdf / .png
# Title inside image: "Fig 5: Multi-Seed Robustness Across Independent Seeds"
# -------------------------------------------------------------------------
print("Generating Fig 5...")
table7_path = proj / "reports" / "tables" / "table7_multi_seed_robustness.csv"
df7 = pd.read_csv(table7_path)

seeds = ["Seed 42", "Seed 123", "Seed 999", "Mean +/- Std"]
s42 = float(df7[df7["Seed"] == "42"]["ROC_AUC"].values[0])
s123 = float(df7[df7["Seed"] == "123"]["ROC_AUC"].values[0])
s999 = float(df7[df7["Seed"] == "999"]["ROC_AUC"].values[0])

mean_str = df7[df7["Seed"] == "Mean +/- Std"]["ROC_AUC"].values[0]
mean_val = float(mean_str.split("+/-")[0].strip())
std_val = float(mean_str.split("+/-")[1].strip())

auc_vals = [s42, s123, s999, mean_val]
auc_errs = [0, 0, 0, std_val]
colors = ["#94a3b8", "#94a3b8", "#94a3b8", "#4338ca"]

fig5, ax5 = plt.subplots(figsize=(6.8, 4.2), dpi=300)
ax5.bar(seeds, auc_vals, yerr=auc_errs, capsize=5, color=colors, edgecolor="black", width=0.55)
ax5.set_ylim(0.65, 0.75)
ax5.set_ylabel("Test ROC-AUC", fontweight="bold")
ax5.set_title("Fig 5: Multi-Seed Robustness Across Independent Seeds", fontweight="bold", fontsize=10, pad=10)

for i, v in enumerate(auc_vals):
    ax5.text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
ax5.grid(axis="y", linestyle="--", alpha=0.5)
fig5.tight_layout()

fig5.savefig(paper_dir / "Fig5_MultiSeed_Robustness.png", dpi=300)
fig5.savefig(paper_dir / "Fig5_MultiSeed_Robustness.pdf")
plt.close(fig5)

print("All figures in paper_final_figures regenerated successfully with matching figure numbers.")
