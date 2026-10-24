"""
Student Knowledge-State Estimation and Predictive Models.

Includes:
1. KnowledgeStateEstimator: Recency-weighted mastery profiling with Bayesian smoothing.
2. BaselineLogisticPredictor: Linear baseline using mastery summaries and item bias.
3. GraphNeuralPredictor: Graph neural network propagating mastery signals along prerequisite edges.
"""
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeStateEstimator:
    """
    Estimates student mastery across all knowledge components.
    
    Weights recent attempts more heavily via exponential decay and applies
    Bayesian smoothing so sparse interaction histories remain well-behaved in [0, 1].
    """

    def __init__(self, num_concepts: int = 1175, prior_mastery: float = 0.5, decay_rate: float = 0.95):
        self.num_concepts = num_concepts
        self.prior_mastery = prior_mastery
        self.decay_rate = decay_rate

    def estimate_mastery(
        self,
        concept_sequence: np.ndarray,
        response_sequence: np.ndarray,
        target_kc: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes a mastery vector for a single student.
        
        Only uses interactions occurring strictly prior to the evaluation point
        to prevent future data leakage.
        """
        mastery = np.full(self.num_concepts, self.prior_mastery, dtype=np.float32)
        counts = np.zeros(self.num_concepts, dtype=np.int32)
        weighted_success = np.zeros(self.num_concepts, dtype=np.float32)
        weight_sum = np.zeros(self.num_concepts, dtype=np.float32)

        n = len(concept_sequence)
        for i in range(n):
            c = concept_sequence[i]
            r = response_sequence[i]
            if 0 <= c < self.num_concepts and (r == 0 or r == 1):
                # More recent practice attempts get higher weight
                recency_weight = math.pow(self.decay_rate, n - 1 - i)
                weighted_success[c] += r * recency_weight
                weight_sum[c] += recency_weight
                counts[c] += 1

        for c in range(self.num_concepts):
            if counts[c] > 0:
                # Bayesian smoothing: blends observed success rate with the 0.5 prior
                mastery[c] = (weighted_success[c] + self.prior_mastery * 2.0) / (weight_sum[c] + 2.0)

        return mastery

class BaselineLogisticPredictor(nn.Module):
    """
    Logistic regression baseline for predicting target concept success.
    
    Uses three features (target mastery, average prerequisite mastery, and practice history length)
    combined with a learned per-concept difficulty bias.
    """

    def __init__(self, num_concepts: int = 1175):
        super().__init__()
        self.num_concepts = num_concepts
        # Learned concept difficulty parameter (bias term)
        self.kc_bias = nn.Embedding(num_concepts, 1)
        # Weights for [target_mastery, prereq_mean_mastery, total_attempts]
        self.linear = nn.Linear(3, 1)

    def forward(
        self,
        target_kc: torch.Tensor,
        target_mastery: torch.Tensor,
        prereq_mastery: torch.Tensor,
        history_len: torch.Tensor
    ) -> torch.Tensor:
        """Predicts probability of a correct response on target_kc."""
        bias = self.kc_bias(target_kc).squeeze(-1) # (B,)
        features = torch.stack([target_mastery, prereq_mastery, history_len], dim=-1) # (B, 3)
        logit = self.linear(features).squeeze(-1) + bias
        return torch.sigmoid(logit)

class GraphNeuralPredictor(nn.Module):
    """
    Graph Neural Network that propagates concept mastery across prerequisite edges.
    
    Uses standard sparse/dense matrix multiplications in PyTorch to ensure
    dependable execution on CPU without extra external graph library dependencies.
    """

    def __init__(
        self,
        num_concepts: int = 1175,
        embedding_dim: int = 32,
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.kc_embeddings = nn.Embedding(num_concepts, embedding_dim)
        # Combines scalar mastery value with learned concept embedding
        self.input_proj = nn.Linear(1 + embedding_dim, hidden_dim)

        # Graph propagation layers (GCN-style message passing on normalized adjacency)
        self.gcn_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_w2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(
        self,
        student_mastery: torch.Tensor, # (B, num_concepts)
        target_kcs: torch.Tensor,      # (B,)
        adj_matrix: torch.Tensor       # (num_concepts, num_concepts) normalized adjacency
    ) -> torch.Tensor:
        """
        Forward pass:
        1. Embeds each concept with its static identity and dynamic student mastery.
        2. Propagates messages across the prerequisite graph.
        3. Predicts response correctness on target_kcs.
        """
        batch_size = student_mastery.size(0)

        # Static KC embeddings: (num_concepts, embedding_dim)
        all_kc_ids = torch.arange(self.num_concepts, device=student_mastery.device)
        kc_emb = self.kc_embeddings(all_kc_ids) # (N, E)

        # Broadcast to batch: (B, N, E)
        kc_emb_b = kc_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # Mastery input: (B, N, 1)
        m_in = student_mastery.unsqueeze(-1)
        # Combine: (B, N, 1 + E)
        x = torch.cat([m_in, kc_emb_b], dim=-1)
        h = F.relu(self.input_proj(x)) # (B, N, H)

        # Layer 1 message passing: H' = A @ H @ W
        # adj_matrix is (N, N)
        h1 = torch.matmul(adj_matrix, h) # (B, N, H)
        h1 = F.relu(self.gcn_w1(h1))
        h1 = self.dropout(h1)

        # Layer 2 message passing
        h2 = torch.matmul(adj_matrix, h1)
        h2 = F.relu(self.gcn_w2(h2)) + h1 # Residual connection

        # Extract representation for the specific target concept for each batch item
        # target_kcs is (B,)
        batch_indices = torch.arange(batch_size, device=target_kcs.device)
        target_repr = h2[batch_indices, target_kcs] # (B, H)

        logits = self.out_head(target_repr).squeeze(-1) # (B,)
        return torch.sigmoid(logits)
