import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

proj = Path("C:/Users/shyam/Downloads/Counterfactual Learning Path/CFLP_Project")

# 1. Regenerate Fig 5 from test_predictions.npz
preds = np.load(proj / "reports" / "evidence_final_20261111" / "predictions" / "test_predictions.npz")
y_test = preds["y_true"]
p_gnn = preds["gnn_probs"]
p_mlp = preds["mlp_probs"]
p_baseline = preds["baseline_probs"]

# Load ROC-AUC values from metrics_by_model.csv
metrics_df = pd.read_csv(proj / "reports" / "evidence_final_20261111" / "metrics_by_model.csv")
# Proposed GNN (Opt Tau), Baseline Logistic, Ablation No-Graph MLP
gnn_auc = float(metrics_df[metrics_df["Model"].str.contains("Proposed GNN")]["ROC_AUC"].values[0])
base_auc = float(metrics_df[metrics_df["Model"].str.contains("Baseline")]["ROC_AUC"].values[0])
mlp_auc = float(metrics_df[metrics_df["Model"].str.contains("No-Graph MLP")]["ROC_AUC"].values[0])

fig5, ax5 = plt.subplots(figsize=(6.8, 5.2), dpi=300)
fpr_g, tpr_g, _ = roc_curve(y_test, p_gnn)
fpr_m, tpr_m, _ = roc_curve(y_test, p_mlp)
fpr_b, tpr_b, _ = roc_curve(y_test, p_baseline)

ax5.plot(fpr_g, tpr_g, label=f"Proposed GNN (AUC = {gnn_auc:.4f})", color="#4338ca", lw=2.2)
ax5.plot(fpr_b, tpr_b, label=f"Baseline Logistic (AUC = {base_auc:.4f})", color="#64748b", linestyle="--", lw=1.8)
ax5.plot(fpr_m, tpr_m, label=f"Ablation No-Graph MLP (AUC = {mlp_auc:.4f})", color="#0284c7", linestyle="-.", lw=1.8)
ax5.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance (AUC = 0.5000)")
ax5.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
ax5.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
ax5.set_title("Figure 5: Receiver Operating Characteristic (ROC) on Held-Out Test Set", fontweight="bold", fontsize=10, pad=10)
ax5.legend(loc="lower right")
ax5.grid(True, linestyle="--", alpha=0.5)
fig5.tight_layout()

for out_dir in [proj / "reports" / "figures", proj / "reports" / "figures_final", proj / "reports" / "evidence_final_20261111" / "figures"]:
    fig5.savefig(out_dir / "fig5_roc_curves.png", dpi=300)
    fig5.savefig(out_dir / "fig5_roc_curves.pdf")
print("Saved regenerated Fig 5 (PNG & PDF)")
plt.close(fig5)

# 2. Regenerate Sensitivity Figure from table_sensitivity_analysis.csv
sens_df = pd.read_csv(proj / "reports" / "tables" / "table_sensitivity_analysis.csv")
intervention_levels = [0.60, 0.75, 0.85, 0.95]

plt.figure(figsize=(8.5, 5), dpi=300)
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
plt.title(r"Figure 11: Counterfactual Prerequisite Impact Sensitivity Across Intervention Levels ($\theta$)", fontsize=10, fontweight="bold", pad=12)
plt.xlabel(r"Counterfactual Target Intervention Mastery ($\theta$)", fontsize=10, fontweight="bold")
plt.ylabel("Estimated Target Probability Gain (%)", fontsize=10, fontweight="bold")
plt.xticks(intervention_levels, [f"{x:.2f}" for x in intervention_levels])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)
plt.tight_layout()

for out_dir in [proj / "reports" / "figures", proj / "reports" / "figures_final", proj / "reports" / "evidence_final_20261111" / "figures"]:
    plt.savefig(out_dir / "counterfactual_sensitivity.png", dpi=300)
    plt.savefig(out_dir / "counterfactual_sensitivity.pdf")
    if out_dir == proj / "reports" / "figures_final":
        plt.savefig(out_dir / "fig11_counterfactual_sensitivity.png", dpi=300)
        plt.savefig(out_dir / "fig11_counterfactual_sensitivity.pdf")
print("Saved regenerated Sensitivity Figure (PNG & PDF)")
plt.close()
