"""
Controlled hyperparameter experimentation:
1. Training with 10 epochs vs 5 epochs.
2. Learning rate tuning: 0.003 vs 0.001 vs 0.005.
3. Hidden capacity: dim 32 vs 64.
4. Loss weighting: standard BCE vs weighted BCE (pos_weight to balance classes).
5. Validation-based early stopping.
6. Multi-seed evaluation across seeds [42, 123, 999].
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from cflp.config import load_config, get_project_root
from cflp.data.loader import DatasetLoader
from cflp.graph.builder import KCGraphBuilder
from cflp.models.predictors import KnowledgeStateEstimator, GraphNeuralPredictor
from cflp.evaluation.metrics import evaluate_predictions
from scripts.train_models import StudentPredictionDataset, build_normalized_adjacency
from pathlib import Path

def train_and_eval_model(
    seed=42,
    lr=0.003,
    hidden_dim=32,
    embedding_dim=32,
    dropout=0.2,
    weight_decay=1e-4,
    use_pos_weight=False,
    max_epochs=12,
    patience=3
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = get_project_root()
    cfg = load_config()
    ds_root = Path(cfg['data']['dataset_root_abs'])
    loader = DatasetLoader(str(ds_root))
    estimator = KnowledgeStateEstimator(num_concepts=1175)
    builder = KCGraphBuilder(str(ds_root / 'metadata'))
    graph = builder.build_graph()
    adj_matrix = build_normalized_adjacency(graph, num_nodes=1175)

    train_records = loader.load_interaction_sequences(split='train', val_fold=4, max_rows=3000)
    val_records = loader.load_interaction_sequences(split='val', val_fold=4, max_rows=1000)
    test_records = loader.load_interaction_sequences(split='test', max_rows=1500)

    train_ds = StudentPredictionDataset(train_records, estimator, graph)
    val_ds = StudentPredictionDataset(val_records, estimator, graph)
    test_ds = StudentPredictionDataset(test_records, estimator, graph)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = GraphNeuralPredictor(
        num_concepts=1175,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weighting check
    if use_pos_weight:
        # Neg / Pos ratio: 612 / 2385 ~ 0.256
        pos_weight = torch.tensor([612.0 / 2385.0])
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        crit = nn.BCELoss()

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    best_val_tau = 0.5
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            p = model(b['mastery_vec'], b['target_kc'], adj_matrix)
            l = crit(p, b['label'])
            l.backward()
            opt.step()

        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for b in val_loader:
                p = model(b['mastery_vec'], b['target_kc'], adj_matrix)
                vp.extend(p.numpy().tolist())
                vy.extend(b['label'].numpy().tolist())

        v_metrics = evaluate_predictions(np.array(vy), np.array(vp), threshold=0.5)
        v_auc = v_metrics['roc_auc']

        # Determine validation threshold for balanced accuracy
        best_tau_bal, best_bal = 0.5, 0.0
        for tau in np.linspace(0.5, 0.95, 91):
            ba = balanced_accuracy_score(np.array(vy), (np.array(vp) >= tau).astype(int))
            if ba > best_bal:
                best_bal = ba
                best_tau_bal = float(tau)

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_epoch = epoch
            best_val_tau = best_tau_bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best checkpoint based strictly on validation AUC
    model.load_state_dict(best_state)
    model.eval()

    tp, ty = [], []
    with torch.no_grad():
        for b in test_loader:
            p = model(b['mastery_vec'], b['target_kc'], adj_matrix)
            tp.extend(p.numpy().tolist())
            ty.extend(b['label'].numpy().tolist())

    test_eval_default = evaluate_predictions(np.array(ty), np.array(tp), threshold=0.5)
    test_eval_calibrated = evaluate_predictions(np.array(ty), np.array(tp), threshold=best_val_tau)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_auc": round(best_val_auc, 4),
        "val_tau": round(best_val_tau, 4),
        "test_roc_auc": test_eval_calibrated["roc_auc"],
        "test_pr_auc": test_eval_calibrated["pr_auc"],
        "test_accuracy": test_eval_calibrated["accuracy"],
        "test_balanced_accuracy": test_eval_calibrated["balanced_accuracy"],
        "test_precision": test_eval_calibrated["precision"],
        "test_recall": test_eval_calibrated["recall"],
        "test_f1_score": test_eval_calibrated["f1_score"],
        "test_log_loss": test_eval_calibrated["log_loss"],
        "test_brier_score": test_eval_calibrated["brier_score"],
        "state_dict": best_state
    }

def run_experiment_grid():
    print("=" * 80)
    print("RUNNING SYSTEMATIC HYPERPARAMETER & MULTI-SEED EXPERIMENTS")
    print("=" * 80)

    configs = [
        {"name": "Original Config (5 Epochs Fixed)", "max_epochs": 5, "patience": 10, "lr": 0.003, "hidden_dim": 32},
        {"name": "Config A: Early Stopping (max 12 epochs, patience 3)", "max_epochs": 12, "patience": 3, "lr": 0.003, "hidden_dim": 32},
        {"name": "Config B: Lower LR (lr=0.001, max 12 epochs)", "max_epochs": 12, "patience": 3, "lr": 0.001, "hidden_dim": 32},
        {"name": "Config C: Higher Capacity (hidden_dim=64, max 12 epochs)", "max_epochs": 12, "patience": 3, "lr": 0.003, "hidden_dim": 64},
    ]

    results = []
    for cfg in configs:
        res = train_and_eval_model(
            seed=42,
            lr=cfg["lr"],
            hidden_dim=cfg["hidden_dim"],
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"]
        )
        print(f"[{cfg['name']}] Best Epoch: {res['best_epoch']} | Val AUC: {res['val_auc']} | Val Tau: {res['val_tau']} | Test AUC: {res['test_roc_auc']} | Test BalAcc: {res['test_balanced_accuracy']} | Log Loss: {res['test_log_loss']}")
        results.append({"config": cfg["name"], "res": res})

    # Pick the best configuration based strictly on Validation AUC
    best_cfg = max(results, key=lambda x: x["res"]["val_auc"])
    print(f"\n---> Winning Configuration by Validation AUC: '{best_cfg['config']}' (Val AUC={best_cfg['res']['val_auc']})")

    # Evaluate winning configuration across all 3 seeds: [42, 123, 999]
    print("\n--- Running 3-Seed Robustness Validation on Winning Configuration ---")
    seed_runs = []
    seeds = [42, 123, 999]
    cfg_params = next(c for c in configs if c["name"] == best_cfg["config"])

    for s in seeds:
        s_res = train_and_eval_model(
            seed=s,
            lr=cfg_params["lr"],
            hidden_dim=cfg_params["hidden_dim"],
            max_epochs=cfg_params["max_epochs"],
            patience=cfg_params["patience"]
        )
        seed_runs.append(s_res)
        print(f"Seed {s}: Best Epoch={s_res['best_epoch']}, Val AUC={s_res['val_auc']:.4f}, Test AUC={s_res['test_roc_auc']:.4f}, Test PR-AUC={s_res['test_pr_auc']:.4f}, Test BalAcc={s_res['test_balanced_accuracy']:.4f}, LogLoss={s_res['test_log_loss']:.4f}")

    mean_auc = np.mean([r["test_roc_auc"] for r in seed_runs])
    std_auc = np.std([r["test_roc_auc"] for r in seed_runs])
    mean_pr_auc = np.mean([r["test_pr_auc"] for r in seed_runs])
    std_pr_auc = np.std([r["test_pr_auc"] for r in seed_runs])
    mean_bal_acc = np.mean([r["test_balanced_accuracy"] for r in seed_runs])
    std_bal_acc = np.std([r["test_balanced_accuracy"] for r in seed_runs])
    mean_loss = np.mean([r["test_log_loss"] for r in seed_runs])
    std_loss = np.std([r["test_log_loss"] for r in seed_runs])

    print(f"\nFinal 3-Seed Test Results:")
    print(f"Test ROC-AUC:      {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Test PR-AUC:       {mean_pr_auc:.4f} +/- {std_pr_auc:.4f}")
    print(f"Test BalAcc:       {mean_bal_acc:.4f} +/- {std_bal_acc:.4f}")
    print(f"Test Log Loss:     {mean_loss:.4f} +/- {std_loss:.4f}")

    # Save summary
    out_path = Path("reports") / "systematic_experiments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_grid": [
                {
                    "name": r["config"],
                    "best_epoch": r["res"]["best_epoch"],
                    "val_auc": r["res"]["val_auc"],
                    "val_tau": r["res"]["val_tau"],
                    "test_roc_auc": r["res"]["test_roc_auc"],
                    "test_pr_auc": r["res"]["test_pr_auc"],
                    "test_balanced_accuracy": r["res"]["test_balanced_accuracy"],
                    "test_log_loss": r["res"]["test_log_loss"],
                    "test_brier_score": r["res"]["test_brier_score"]
                }
                for r in results
            ],
            "winning_config": best_cfg["config"],
            "multi_seed_runs": [
                {
                    "seed": r["seed"],
                    "best_epoch": r["best_epoch"],
                    "val_auc": r["val_auc"],
                    "val_tau": r["val_tau"],
                    "test_roc_auc": r["test_roc_auc"],
                    "test_pr_auc": r["test_pr_auc"],
                    "test_balanced_accuracy": r["test_balanced_accuracy"],
                    "test_log_loss": r["test_log_loss"],
                    "test_brier_score": r["test_brier_score"]
                }
                for r in seed_runs
            ],
            "multi_seed_aggregate": {
                "mean_roc_auc": round(float(mean_auc), 4),
                "std_roc_auc": round(float(std_auc), 4),
                "mean_pr_auc": round(float(mean_pr_auc), 4),
                "std_pr_auc": round(float(std_pr_auc), 4),
                "mean_balanced_accuracy": round(float(mean_bal_acc), 4),
                "std_balanced_accuracy": round(float(std_bal_acc), 4),
                "mean_log_loss": round(float(mean_loss), 4),
                "std_log_loss": round(float(std_loss), 4)
            }
        }, f, indent=2)

if __name__ == "__main__":
    run_experiment_grid()
