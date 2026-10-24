"""
Counterfactual Impact Engine

Simulates hypothetical mastery improvements on prerequisite knowledge components
and estimates downstream impact on target concept success:
    I_i = M_target(CF_i) - M_target(factual)
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import networkx as nx

class CounterfactualImpactEngine:
    """Estimates how target concept performance changes when prerequisite mastery improves."""

    def __init__(
        self,
        model: torch.nn.Module,
        graph: nx.DiGraph,
        adj_matrix: torch.Tensor,
        device: str = "cpu"
    ):
        self.model = model
        self.graph = graph
        self.adj_matrix = adj_matrix
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def select_candidates(
        self,
        target_kc: int,
        student_mastery: np.ndarray,
        max_candidates: int = 10,
        weakness_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Finds candidate prerequisites for a target concept, prioritizing weaker concepts
        where remediation can provide the most benefit.
        """
        if not self.graph.has_node(target_kc):
            return []

        # Look at both immediate and two-step prerequisites in the graph
        predecessors = set(self.graph.predecessors(target_kc))
        two_hop = set()
        for p in predecessors:
            two_hop.update(self.graph.predecessors(p))
        candidates_set = (predecessors | two_hop) - {target_kc}

        candidate_records = []
        for cid in candidates_set:
            current_m = float(student_mastery[cid])
            is_direct = cid in predecessors
            # Distance / relationship
            rel = "direct_prerequisite" if is_direct else "indirect_prerequisite (2-hop)"
            candidate_records.append({
                "kc_id": cid,
                "current_mastery": current_m,
                "relationship": rel,
                "is_weak": current_m < weakness_threshold
            })

        # Sort: weakest concepts first
        candidate_records.sort(key=lambda x: (not x["is_weak"], x["current_mastery"]))
        return candidate_records[:max_candidates]

    def compute_counterfactual_impacts(
        self,
        target_kc: int,
        student_mastery: np.ndarray,
        candidates: Optional[List[int]] = None,
        intervention_level: float = 0.85,
        enforce_improvement: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Calculates counterfactual impact (I_i = CF - Factual) for each candidate prerequisite.
        
        When enforce_improvement is True, ensures the simulated intervention actually represents
        a skill boost above the student's current mastery rather than an artificial reduction.
        """
        if candidates is None:
            cand_records = self.select_candidates(target_kc, student_mastery)
            candidate_ids = [c["kc_id"] for c in cand_records]
            cand_meta = {c["kc_id"]: c for c in cand_records}
        else:
            candidate_ids = candidates
            cand_meta = {
                cid: {
                    "kc_id": cid,
                    "current_mastery": float(student_mastery[cid]),
                    "relationship": "user_specified"
                }
                for cid in candidate_ids
            }

        if not candidate_ids:
            return []

        # 1. Evaluate baseline (factual) probability without any intervention
        with torch.no_grad():
            m_factual_t = torch.tensor(student_mastery, dtype=torch.float32, device=self.device).unsqueeze(0)
            target_t = torch.tensor([target_kc], dtype=torch.long, device=self.device)
            adj = self.adj_matrix.to(self.device)

            factual_score = float(self.model(m_factual_t, target_t, adj).item())

            # 2. Evaluate counterfactual scenarios in a single batch for efficiency
            B = len(candidate_ids)
            m_batch = np.tile(student_mastery, (B, 1))
            actual_interventions = []
            intervention_types = []

            for idx, cid in enumerate(candidate_ids):
                current_m = float(student_mastery[cid])
                target_val = float(intervention_level)

                if current_m >= 1.0:
                    # Student has already achieved complete mastery on this concept
                    target_val = 1.0
                    itype = "at_ceiling"
                elif enforce_improvement and target_val <= current_m:
                    # When requested level is below current mastery, apply a positive delta instead
                    target_val = min(1.0, current_m + 0.10)
                    itype = "adjusted_improvement" if target_val > current_m else "at_ceiling"
                else:
                    itype = "direct_improvement" if target_val > current_m else "skill_reduction"

                actual_interventions.append(target_val)
                intervention_types.append(itype)
                m_batch[idx, cid] = target_val

            m_cf_t = torch.tensor(m_batch, dtype=torch.float32, device=self.device)
            target_batch_t = torch.full((B,), target_kc, dtype=torch.long, device=self.device)

            cf_scores = self.model(m_cf_t, target_batch_t, adj).cpu().numpy()

        results = []
        for idx, cid in enumerate(candidate_ids):
            cf_val = float(cf_scores[idx])
            impact = cf_val - factual_score
            meta = cand_meta.get(cid, {})

            results.append({
                "kc_id": cid,
                "current_mastery": float(student_mastery[cid]),
                "requested_intervention_level": float(intervention_level),
                "actual_intervention_level": round(actual_interventions[idx], 4),
                "intervention_type": intervention_types[idx],
                "factual_target_prediction": round(factual_score, 4),
                "counterfactual_target_prediction": round(cf_val, 4),
                "estimated_impact": round(impact, 4),
                "relationship": meta.get("relationship", "candidate_relationship"),
                "causal_disclaimer": "Model-based counterfactual estimate under simulated feature intervention, not verified experimental causal effect."
            })

        # Rank candidates by estimated impact descending
        results.sort(key=lambda x: x["estimated_impact"], reverse=True)
        return results
