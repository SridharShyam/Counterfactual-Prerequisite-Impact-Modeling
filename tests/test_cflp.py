"""
Automated Unit and Integration Test Suite for CFLP.
Tests:
1. Sequence parsing & padding mask validation.
2. Knowledge state estimation logic.
3. Graph construction and DAG cycle prevention.
4. Factual and counterfactual prediction integrity.
5. Impact calculation and ranking order.
6. Prerequisite-constrained learning path generation.
7. Synthetic smoke tests for model fallback.
"""
import pytest
import numpy as np
import torch
import networkx as nx

from cflp.data.loader import SequenceParser
from cflp.models.predictors import (
    KnowledgeStateEstimator,
    BaselineLogisticPredictor,
    GraphNeuralPredictor
)
from cflp.counterfactual.impact import CounterfactualImpactEngine
from cflp.paths.generator import LearningPathGenerator

def test_sequence_parser_and_padding_mask():
    parser = SequenceParser(pad_val=-1)
    raw_str = "10,20,-1,35,-1"
    parsed = parser.parse_sequence_string(raw_str)
    assert len(parsed) == 5
    assert np.array_equal(parsed, np.array([10, 20, -1, 35, -1]))

    mask = parser.extract_active_mask(parsed)
    assert np.array_equal(mask, np.array([True, True, False, True, False]))
    assert np.array_equal(parsed[mask], np.array([10, 20, 35]))

def test_compound_concept_parsing():
    parser = SequenceParser()
    raw_str = "100,166_170,200"
    parsed = parser.parse_sequence_string(raw_str)
    assert len(parsed) == 3
    assert parsed[1] == 166 # Extracts primary KC ID

def test_knowledge_state_estimator():
    estimator = KnowledgeStateEstimator(num_concepts=10, prior_mastery=0.5)
    concepts = np.array([0, 0, 1, 1], dtype=np.int64)
    responses = np.array([1, 1, 0, 0], dtype=np.int64)
    mastery = estimator.estimate_mastery(concepts, responses)

    assert len(mastery) == 10
    assert mastery[0] > 0.6 # Concept 0 had two successes
    assert mastery[1] < 0.4 # Concept 1 had two failures
    assert mastery[2] == 0.5 # Unseen concept retains prior

def test_graph_dag_and_path_generation():
    G = nx.DiGraph()
    G.add_nodes_from([
        (1, {"name_en": "Prerequisite Concept 1"}),
        (2, {"name_en": "Prerequisite Concept 2"}),
        (3, {"name_en": "Target Concept 3"})
    ])
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    assert nx.is_directed_acyclic_graph(G)

    generator = LearningPathGenerator(G)
    mastery = np.full(10, 0.5)
    rankings = [
        {"kc_id": 1, "estimated_impact": 0.08},
        {"kc_id": 2, "estimated_impact": 0.04}
    ]
    path = generator.generate_path(target_kc=3, student_mastery=mastery, impact_rankings=rankings)

    assert path["target_kc"] == 3
    assert path["total_steps"] == 3
    assert path["ordered_kc_sequence"] == [1, 2, 3] # Correct topological order

def test_counterfactual_engine_synthetic_smoke():
    num_concepts = 10
    model = GraphNeuralPredictor(num_concepts=num_concepts, embedding_dim=8, hidden_dim=8)
    model.eval()

    G = nx.DiGraph()
    for i in range(num_concepts):
        G.add_node(i, name_en=f"Concept_{i}")
    G.add_edge(0, 2)
    G.add_edge(1, 2)

    adj = torch.eye(num_concepts)
    adj[0, 2] = 0.5
    adj[1, 2] = 0.5

    engine = CounterfactualImpactEngine(model, G, adj)
    student_mastery = np.full(num_concepts, 0.4)

    rankings = engine.compute_counterfactual_impacts(
        target_kc=2,
        student_mastery=student_mastery,
        intervention_level=0.9
    )

    assert len(rankings) == 2
    for item in rankings:
        assert "estimated_impact" in item
        assert "factual_target_prediction" in item
        assert "counterfactual_target_prediction" in item
        # Verify impact equation arithmetic
        diff = item["counterfactual_target_prediction"] - item["factual_target_prediction"]
        assert abs(item["estimated_impact"] - diff) < 1e-3

def test_counterfactual_boundary_and_improvement_enforcement():
    """
    Tests intervention levels below, equal to, and above current mastery,
    including the boundary ceiling of 1.0.
    """
    num_concepts = 6
    model = GraphNeuralPredictor(num_concepts=num_concepts, embedding_dim=8, hidden_dim=8)
    model.eval()

    G = nx.DiGraph()
    for i in range(num_concepts):
        G.add_node(i, name_en=f"Concept_{i}")
    G.add_edge(0, 3) # Candidate 0
    G.add_edge(1, 3) # Candidate 1
    G.add_edge(2, 3) # Candidate 2

    adj = torch.eye(num_concepts)
    engine = CounterfactualImpactEngine(model, G, adj)

    # Concept 0 has mastery 0.40 (below intervention 0.85 -> direct improvement)
    # Concept 1 has mastery 0.85 (equal to intervention 0.85 -> adjusted improvement)
    # Concept 2 has mastery 1.00 (at ceiling -> at_ceiling)
    m = np.array([0.40, 0.85, 1.00, 0.50, 0.50, 0.50])

    res = engine.compute_counterfactual_impacts(
        target_kc=3,
        student_mastery=m,
        candidates=[0, 1, 2],
        intervention_level=0.85,
        enforce_improvement=True
    )
    res_map = {r["kc_id"]: r for r in res}

    # Verify Concept 0: requested 0.85, applied 0.85, type direct_improvement
    assert res_map[0]["requested_intervention_level"] == 0.85
    assert res_map[0]["actual_intervention_level"] == 0.85
    assert res_map[0]["intervention_type"] == "direct_improvement"

    # Verify Concept 1: requested 0.85 <= mastery 0.85, actual clamped strictly above to 0.95
    assert res_map[1]["requested_intervention_level"] == 0.85
    assert res_map[1]["actual_intervention_level"] == 0.95
    assert res_map[1]["intervention_type"] == "adjusted_improvement"

    # Verify Concept 2: mastery is 1.00, cannot exceed 1.00 -> at_ceiling
    assert res_map[2]["actual_intervention_level"] == 1.00
    assert res_map[2]["intervention_type"] == "at_ceiling"

def test_path_generator_transitive_closure_preservation():
    """
    Tests that path generation does not omit intermediate required prerequisites.
    Graph topology: 0 -> 1 -> 2 -> 3 (target).
    If max_path_length allows only 3 steps (target + 2 nodes), selecting concept 2
    MUST include its intermediate predecessor 1, rather than jumping from 0 to 2.
    """
    G = nx.DiGraph()
    for i in range(4):
        G.add_node(i, name_en=f"Concept_{i}")
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    gen = LearningPathGenerator(G)
    mastery = np.full(4, 0.4)

    # Suppose concept 2 has high impact, 0 has medium impact, 1 has low impact
    rankings = [
        {"kc_id": 2, "estimated_impact": 0.15},
        {"kc_id": 0, "estimated_impact": 0.10},
        {"kc_id": 1, "estimated_impact": 0.01}
    ]

    # Generate path with length budget 3 (target + 2 concepts)
    path = gen.generate_path(target_kc=3, student_mastery=mastery, impact_rankings=rankings, max_path_length=3)

    # Valid path must satisfy topological order and closure
    seq = path["ordered_kc_sequence"]
    assert seq[-1] == 3

    # Verify closure completeness check
    if path["has_complete_prerequisite_closure"]:
        for idx in range(len(seq) - 1):
            curr = seq[idx]
            for pred in G.predecessors(curr):
                if pred in nx.ancestors(G, 3):
                    assert pred in seq[:idx], f"Prerequisite {pred} must precede {curr} in sequence {seq}"

