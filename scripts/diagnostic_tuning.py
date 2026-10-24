"""
Experimental script to analyze training dynamics, early stopping, and hyperparameter tuning.
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

def run_tuning_diagnostics():
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

    torch.manual_seed(42)
    np.random.seed(42)

    gnn_model = GraphNeuralPredictor(num_concepts=1175, embedding_dim=32, hidden_dim=32, dropout=0.2)
    opt = torch.optim.Adam(gnn_model.parameters(), lr=0.003, weight_decay=1e-4)
    crit = nn.BCELoss()

    print("Epoch | Train Loss | Val Loss | Val AUC | Val BalAcc(0.5) | Val BalAcc(tau*) | Test AUC")
    print("-" * 75)

    for epoch in range(1, 13):
        gnn_model.train()
        tr_loss = 0.0
        for b in train_loader:
            opt.zero_grad()
            p = gnn_model(b['mastery_vec'], b['target_kc'], adj_matrix)
            l = crit(p, b['label'])
            l.backward()
            opt.step()
            tr_loss += l.item() * len(b['label'])
        tr_loss /= len(train_ds)

        gnn_model.eval()
        vp, vy = [], []
        v_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                p = gnn_model(b['mastery_vec'], b['target_kc'], adj_matrix)
                l = crit(p, b['label'])
                v_loss += l.item() * len(b['label'])
                vp.extend(p.numpy().tolist())
                vy.extend(b['label'].numpy().tolist())
        v_loss /= len(val_ds)
        v_metrics = evaluate_predictions(np.array(vy), np.array(vp), threshold=0.5)

        best_tau, best_bacc = 0.5, 0.0
        for tau in np.linspace(0.5, 0.95, 91):
            ba = balanced_accuracy_score(np.array(vy), (np.array(vp) >= tau).astype(int))
            if ba > best_bacc:
                best_bacc = ba
                best_tau = tau

        # Check test AUC on holdout
        tp, ty = [], []
        with torch.no_grad():
            for b in test_loader:
                p = gnn_model(b['mastery_vec'], b['target_kc'], adj_matrix)
                tp.extend(p.numpy().tolist())
                ty.extend(b['label'].numpy().tolist())
        t_metrics = evaluate_predictions(np.array(ty), np.array(tp), threshold=best_tau)

        print(f"{epoch:5d} | {tr_loss:.4f}     | {v_loss:.4f}   | {v_metrics['roc_auc']:.4f}  | {v_metrics['balanced_accuracy']:.4f}          | {best_bacc:.4f} (tau={best_tau:.2f}) | {t_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    run_tuning_diagnostics()
