"""
Personalized Learning Path Generator

Combines curriculum prerequisite constraints with counterfactual impact estimates
and current student mastery to construct an ordered, actionable study path.
"""
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import numpy as np

class LearningPathGenerator:
    """Builds prerequisite-respecting study sequences tailored to a student's weaknesses."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def generate_path(
        self,
        target_kc: int,
        student_mastery: np.ndarray,
        impact_rankings: List[Dict[str, Any]],
        mastery_threshold: float = 0.75,
        max_path_length: int = 5
    ) -> Dict[str, Any]:
        """
        Builds a step-by-step learning path toward target_kc.
        
        Identifies unmastered prerequisites, ranks them by potential impact,
        and applies topological sorting so prerequisites are always studied
        before dependent concepts.
        """
        if not self.graph.has_node(target_kc):
            return {
                "target_kc": target_kc,
                "path": [target_kc],
                "explanation": "Target concept has no prerequisite links in the graph; direct study recommended.",
                "steps": []
            }

        # 1. Identify all foundational prerequisite concepts (ancestors) in the graph
        ancestors = nx.ancestors(self.graph, target_kc)

        # 2. Prioritize concepts where the student is struggling and potential impact is highest
        impact_lookup = {item["kc_id"]: item["estimated_impact"] for item in impact_rankings}

        candidate_scores = []
        for c in ancestors:
            current_m = float(student_mastery[c])
            impact = impact_lookup.get(c, 0.0)

            # Balance mastery gap (60%) with counterfactual lift (40%)
            priority_score = (1.0 - current_m) * 0.6 + max(0.0, impact) * 0.4

            if current_m < mastery_threshold or impact > 0.01:
                candidate_scores.append((c, priority_score, current_m, impact))

        # Sort candidates by priority score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [x[0] for x in candidate_scores]

        # 3. Construct prerequisite closure:
        # A selected concept should NOT omit its intermediate required prerequisites.
        # We start with the highest priority candidate and expand its required ancestor closure.
        selected_nodes = set()
        unreachable_flag = False

        for cid in top_candidates:
            # Ancestors of this candidate within the main graph
            c_anc = nx.ancestors(self.graph, cid)
            needed_nodes = (c_anc | {cid})
            if len(selected_nodes | needed_nodes) + 1 <= max_path_length:
                selected_nodes.update(needed_nodes)
            else:
                # If we cannot include the full prerequisite closure within budget, skip to prevent gaps
                unreachable_flag = True

        # If empty (e.g. even direct predecessor closure exceeds max_path_length),
        # include immediate predecessors up to budget and flag incomplete closure
        if not selected_nodes and top_candidates:
            selected_nodes.add(top_candidates[0])
            unreachable_flag = True

        # 4. Create induced subgraph with selected ancestors + target_kc
        path_nodes = set(selected_nodes) | {target_kc}
        subgraph = self.graph.subgraph(path_nodes).copy()

        # 5. Topological order to preserve prerequisite integrity
        try:
            topo_order = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # Fallback if any unexpected cycle
            topo_order = sorted(list(path_nodes), key=lambda x: (x == target_kc, -impact_lookup.get(x, 0.0)))

        # Ensure target_kc is strictly at the terminus of the sequence
        if target_kc in topo_order:
            topo_order.remove(target_kc)
        topo_order.append(target_kc)

        # 6. Verification of prerequisite closure completeness
        # Check if every node's in-graph predecessors that are in 'ancestors' are present in path
        missing_intermediates = []
        for n in topo_order[:-1]:
            for pred in self.graph.predecessors(n):
                if pred not in path_nodes and pred in ancestors:
                    missing_intermediates.append((pred, n))

        has_complete_closure = (len(missing_intermediates) == 0)

        steps = []
        for idx, cid in enumerate(topo_order):
            is_target = (cid == target_kc)
            cur_m = float(student_mastery[cid])
            imp = impact_lookup.get(cid, 0.0)
            node_data = self.graph.nodes.get(cid, {})

            if is_target:
                reason = "Target Concept: Goal of the personalized learning progression."
            else:
                reason = f"Prerequisite Concept: Current mastery is {cur_m:.1%}. Improves target probability by +{imp:.1%}."

            steps.append({
                "step_number": idx + 1,
                "kc_id": cid,
                "kc_name_en": node_data.get("name_en", f"Concept_{cid}"),
                "kc_name_zh": node_data.get("name_zh", ""),
                "current_mastery": round(cur_m, 4),
                "estimated_impact": round(imp, 4),
                "is_target": is_target,
                "rationale": reason
            })

        return {
            "target_kc": target_kc,
            "target_kc_name_en": self.graph.nodes.get(target_kc, {}).get("name_en", f"Concept_{target_kc}"),
            "total_steps": len(steps),
            "ordered_kc_sequence": topo_order,
            "steps": steps,
            "has_complete_prerequisite_closure": has_complete_closure,
            "missing_intermediate_prerequisites": missing_intermediates,
            "is_reachable_within_length_budget": not unreachable_flag or has_complete_closure,
            "pedagogical_constraint": "Topologically ordered with transitive closure verification to ensure all intermediate prerequisites precede advanced topics."
        }
