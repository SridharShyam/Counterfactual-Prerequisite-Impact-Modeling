import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def regenerate_fig8():
    proj_dir = Path(__file__).resolve().parent.parent
    table_path = proj_dir / "reports" / "tables" / "table7_multi_seed_robustness.csv"
    df = pd.read_csv(table_path)
    
    seeds = ["Seed 42", "Seed 123", "Seed 999", "Mean +/- Std"]
    s42 = float(df[df["Seed"] == "42"]["ROC_AUC"].values[0])
    s123 = float(df[df["Seed"] == "123"]["ROC_AUC"].values[0])
    s999 = float(df[df["Seed"] == "999"]["ROC_AUC"].values[0])
    
    mean_str = df[df["Seed"] == "Mean +/- Std"]["ROC_AUC"].values[0]
    mean_val = float(mean_str.split("+/-")[0].strip())
    std_val = float(mean_str.split("+/-")[1].strip())
    
    auc_vals = [s42, s123, s999, mean_val]
    auc_errs = [0, 0, 0, std_val]
    colors = ["#94a3b8", "#94a3b8", "#94a3b8", "#4338ca"]
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.bar(seeds, auc_vals, yerr=auc_errs, capsize=5, color=colors, edgecolor="black", width=0.55)
    ax.set_ylim(0.65, 0.75)
    ax.set_ylabel("Test ROC-AUC", fontweight="bold")
    ax.set_title("Figure 8: Multi-Seed Robustness Across Independent Seeds", fontweight="bold")
    
    for i, v in enumerate(auc_vals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    # Save to targets
    targets = [
        proj_dir / "reports" / "figures_final" / "fig8_multiseed_robustness",
        proj_dir / "reports" / "evidence_final_20261111" / "figures" / "fig8_multiseed_robustness",
        proj_dir / "reports" / "figures" / "fig8_multiseed_robustness"
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{target}.png", dpi=300)
        fig.savefig(f"{target}.pdf")
        print(f"Saved: {target}.png and .pdf")
    plt.close(fig)

if __name__ == "__main__":
    regenerate_fig8()
