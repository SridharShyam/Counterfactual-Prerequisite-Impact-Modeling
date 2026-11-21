import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

proj = Path("C:/Users/shyam/Downloads/Counterfactual Learning Path/CFLP_Project")
paper_dir = proj / "paper_final_figures"
reports_fig_final = proj / "reports" / "figures_final"
evidence_fig = proj / "reports" / "evidence_final_20261111" / "figures"
reports_fig = proj / "reports" / "figures"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["text.color"] = "#000000"

def draw_fig1():
    # 14 x 4.0 inches gives generous horizontal spacing and clean margins
    fig, ax = plt.subplots(figsize=(14.2, 4.0), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stages = [
        {
            "header": "Student\nInteraction\nData",
            "boxes": ["Questions /\nExercises", "Responses", "Timestamps"],
        },
        {
            "header": "Knowledge\nState Estimation",
            "boxes": ["Estimate\nConcept Mastery", "Identify\nWeak Concepts"],
        },
        {
            "header": "Student\nKnowledge Graph",
            "boxes": ["Student–Concept\nRelations", "Prerequisite\nRelations"],
        },
        {
            "header": "GNN-Based\nKnowledge Prediction",
            "boxes": ["GNN", "Mastery\nPrediction"],
        },
        {
            "header": "Counterfactual\nImpact Analysis",
            "boxes": ["Hypothetical\nIntervention", "Predicted\nImpact"],
        },
        {
            "header": "Prerequisite\nRanking",
            "boxes": ["Impact Score", "Rank\nPrerequisites"],
        },
        {
            "header": "Personalized\nLearning Path",
            "boxes": ["Prerequisite-\nConstrained Path", "Target Concept"],
        },
    ]

    n = len(stages)
    # Balanced layout parameters to ensure zero overlap between dashed containers
    col_w = 0.116
    gap = 0.023
    start_x = (1.0 - (n * col_w + (n - 1) * gap)) / 2.0  # perfectly centered horizontally
    outer_y = 0.06
    outer_h = 0.78

    for i, s in enumerate(stages):
        x = start_x + i * (col_w + gap)
        
        # 1. Outer dashed rounded box
        outer_rect = patches.FancyBboxPatch(
            (x, outer_y), col_w, outer_h,
            boxstyle="round,pad=0.008",
            fc="#ffffff", ec="#000000", lw=1.2, ls="--"
        )
        ax.add_patch(outer_rect)

        # 2. Header text at the top of the container
        header_y = outer_y + outer_h - 0.11
        ax.text(
            x + col_w / 2, header_y, s["header"],
            ha="center", va="center", fontsize=8.8, fontweight="bold",
            linespacing=1.15, color="#000000"
        )

        # 3. Inner solid sub-boxes
        num_boxes = len(s["boxes"])
        avail_bottom = outer_y + 0.035
        avail_top = header_y - 0.080
        total_avail_h = avail_top - avail_bottom
        
        box_gap = 0.022
        inner_w = col_w * 0.85
        inner_x = x + (col_w - inner_w) / 2
        
        inner_h = (total_avail_h - (num_boxes - 1) * box_gap) / num_boxes

        for b_idx, b_text in enumerate(s["boxes"]):
            b_y = avail_top - (b_idx + 1) * inner_h - b_idx * box_gap
            inner_rect = patches.FancyBboxPatch(
                (inner_x, b_y), inner_w, inner_h,
                boxstyle="square,pad=0",
                fc="#ffffff", ec="#000000", lw=1.0, ls="-"
            )
            ax.add_patch(inner_rect)
            ax.text(
                inner_x + inner_w / 2, b_y + inner_h / 2, b_text,
                ha="center", va="center", fontsize=8.0, color="#000000",
                linespacing=1.12
            )

        # 4. Connecting arrow between stages
        if i < n - 1:
            arrow_start_x = x + col_w + 0.003
            arrow_end_x = x + col_w + gap - 0.003
            arrow_y = outer_y + outer_h / 2
            ax.annotate(
                "",
                xy=(arrow_end_x, arrow_y),
                xytext=(arrow_start_x, arrow_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=1.3,
                    mutation_scale=11,
                    color="#000000"
                )
            )

    # Clean centered title matching "Fig 1"
    ax.text(
        0.5, 0.94,
        "Fig 1: Counterfactual Learning Path (C-PrIM) System Architecture & Pipeline",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#000000"
    )

    fig.tight_layout()
    
    # Save targets
    for target_dir in [paper_dir, reports_fig_final, evidence_fig, reports_fig]:
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_dir == paper_dir:
            fig.savefig(target_dir / "Fig1_Architecture.png", dpi=300)
            fig.savefig(target_dir / "Fig1_Architecture.pdf")
        else:
            fig.savefig(target_dir / "fig1_cflp_pipeline_architecture.png", dpi=300)
            fig.savefig(target_dir / "fig1_cflp_pipeline_architecture.pdf")

    plt.close(fig)
    print("Fig 1 refined and saved across all figure repositories.")

if __name__ == "__main__":
    draw_fig1()
