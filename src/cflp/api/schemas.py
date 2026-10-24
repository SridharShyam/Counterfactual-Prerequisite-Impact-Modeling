"""
Pydantic Schemas for C-PrIM REST API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ConceptResponse(BaseModel):
    kc_id: int
    name_en: str
    name_zh: Optional[str] = None
    in_degree: int
    out_degree: int
    direct_prerequisites: List[int] = []

class MasteryEstimateRequest(BaseModel):
    concepts: List[int] = Field(..., description="Chronological sequence of concept IDs")
    responses: List[int] = Field(..., description="Chronological sequence of binary responses (0 or 1)")

class MasteryEstimateResponse(BaseModel):
    num_concepts_evaluated: int
    active_interactions: int
    overall_mean_mastery: float
    mastery_profile: Dict[int, float] = Field(..., description="Map of active KC ID to estimated mastery in [0, 1]")

class PredictRequest(BaseModel):
    student_mastery: Optional[Dict[int, float]] = None
    student_history: Optional[MasteryEstimateRequest] = None
    target_kc: int

class PredictResponse(BaseModel):
    target_kc: int
    target_kc_name_en: str
    predicted_success_probability: float
    model_type: str

class CounterfactualRankRequest(BaseModel):
    target_kc: int
    student_mastery: Optional[Dict[int, float]] = None
    student_history: Optional[MasteryEstimateRequest] = None
    intervention_level: float = Field(0.85, ge=0.0, le=1.0)
    candidate_kcs: Optional[List[int]] = None

class CandidateImpactItem(BaseModel):
    kc_id: int
    kc_name_en: str
    current_mastery: float
    intervention_level: float
    factual_target_prediction: float
    counterfactual_target_prediction: float
    estimated_impact: float
    relationship: str

class CounterfactualRankResponse(BaseModel):
    target_kc: int
    target_kc_name_en: str
    intervention_level: float
    factual_target_prediction: float
    rankings: List[CandidateImpactItem]
    scientific_disclaimer: str

class LearningPathRequest(BaseModel):
    target_kc: int
    student_mastery: Optional[Dict[int, float]] = None
    student_history: Optional[MasteryEstimateRequest] = None
    intervention_level: float = Field(0.85, ge=0.0, le=1.0)
    max_path_length: int = Field(5, ge=1, le=15)

class PathStepItem(BaseModel):
    step_number: int
    kc_id: int
    kc_name_en: str
    current_mastery: float
    estimated_impact: float
    is_target: bool
    rationale: str

class LearningPathResponse(BaseModel):
    target_kc: int
    target_kc_name_en: str
    total_steps: int
    ordered_kc_sequence: List[int]
    steps: List[PathStepItem]
    pedagogical_constraint: str
