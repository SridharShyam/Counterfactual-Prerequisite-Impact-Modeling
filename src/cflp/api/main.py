"""
FastAPI application for CFLP:
Exposes health, concepts metadata, mastery estimation, factual prediction,
counterfactual prerequisite impact ranking, and personalized learning path generation.
"""
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cflp.config import load_config, get_project_root
from cflp.data.loader import DatasetLoader
from cflp.graph.builder import KCGraphBuilder
from cflp.models.predictors import KnowledgeStateEstimator, GraphNeuralPredictor
from cflp.counterfactual.impact import CounterfactualImpactEngine
from cflp.paths.generator import LearningPathGenerator
from cflp.api.schemas import (
    ConceptResponse,
    MasteryEstimateRequest,
    MasteryEstimateResponse,
    PredictRequest,
    PredictResponse,
    CounterfactualRankRequest,
    CounterfactualRankResponse,
    LearningPathRequest,
    LearningPathResponse
)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load datasets, graph, and trained model artifacts
    root = get_project_root()
    cfg = load_config()
    ds_root = Path(cfg["data"]["dataset_root_abs"])
    artifacts_dir = root / "artifacts"

    import logging
    logger = logging.getLogger("cflp.api")

    # Diagnostic check: verify whether external dataset metadata exists
    if not ds_root.exists() or not (ds_root / "metadata").exists():
        logger.warning(
            f"Dataset directory '{ds_root}' not found. Starting API with synthetic test fixtures."
        )
        loader = None
        graph = nx.DiGraph()
        for i in range(10):
            graph.add_node(i, id=i, name_en=f"Synthetic_Concept_{i}", name_zh=f"概念_{i}")
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
    else:
        logger.info(f"Loading dataset from {ds_root} and building prerequisite graph...")
        loader = DatasetLoader(str(ds_root))
        builder = KCGraphBuilder(str(ds_root / "metadata"))
        graph = builder.build_graph()

    from safetensors.torch import load_file

    adj_path = artifacts_dir / "normalized_adjacency.npy"
    gnn_safetensors = artifacts_dir / "gnn_model.safetensors"

    if adj_path.exists() and gnn_safetensors.exists() and graph.number_of_nodes() >= 1175:
        logger.info(f"Loading trained GNN weights from {gnn_safetensors}...")
        adj_np = np.load(adj_path)
        adj_matrix = torch.from_numpy(adj_np).float()
        gnn_model = GraphNeuralPredictor(num_concepts=1175, embedding_dim=32, hidden_dim=32, dropout=0.2)
        state = load_file(str(gnn_safetensors))
        gnn_model.load_state_dict(state)
        gnn_model.eval()
        cf_engine = CounterfactualImpactEngine(gnn_model, graph, adj_matrix)
    else:
        logger.warning("Pre-trained model checkpoint or adjacency matrix not found in artifacts/. Using uninitialized fallback GNN.")
        n_c = max(10, graph.number_of_nodes())
        gnn_model = GraphNeuralPredictor(num_concepts=n_c, embedding_dim=16, hidden_dim=16, dropout=0.0)
        gnn_model.eval()
        adj_matrix = torch.eye(n_c)
        cf_engine = CounterfactualImpactEngine(gnn_model, graph, adj_matrix)

    estimator = KnowledgeStateEstimator(num_concepts=max(10, graph.number_of_nodes()))
    path_generator = LearningPathGenerator(graph)

    SERVICES["loader"] = loader
    SERVICES["estimator"] = estimator
    SERVICES["graph"] = graph
    SERVICES["adj_matrix"] = adj_matrix
    SERVICES["gnn_model"] = gnn_model
    SERVICES["cf_engine"] = cf_engine
    SERVICES["path_generator"] = path_generator
    yield
    SERVICES.clear()

app = FastAPI(
    title="Counterfactual Prerequisite Impact API",
    description="Research prototype backend for counterfactual prerequisite impact modeling and learning path generation.",
    version="0.1.0",
    lifespan=lifespan
)

ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serves the research prototype dashboard interface."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "CFLP Research API online. View /docs for interactive OpenAPI specification."}

SERVICES: Dict[str, Any] = {}

@app.get("/concepts", response_model=List[ConceptResponse])
def list_concepts(
    limit: int = 50,
    offset: int = 0,
    only_with_prerequisites: bool = False
):
    """Returns list of concepts with pagination and prerequisite filtering."""
    graph: nx.DiGraph = SERVICES["graph"]
    loader: DatasetLoader = SERVICES["loader"]
    results = []

    all_nodes = sorted(list(graph.nodes()))
    if only_with_prerequisites:
        all_nodes = [n for n in all_nodes if graph.in_degree(n) > 0]

    paginated = all_nodes[offset : offset + limit]
    for cid in paginated:
        node = graph.nodes[cid]
        preds = list(graph.predecessors(cid))
        results.append(
            ConceptResponse(
                kc_id=cid,
                name_en=node.get("name_en", f"Concept_{cid}"),
                name_zh=node.get("name_zh"),
                in_degree=graph.in_degree(cid),
                out_degree=graph.out_degree(cid),
                direct_prerequisites=preds
            )
        )
    return results

@app.get("/graph/subgraph/{kc_id}")
def get_graph_subgraph(kc_id: int, depth: int = 2):
    """Returns nodes and directed edges for the target concept and its prerequisite hierarchy."""
    graph: nx.DiGraph = SERVICES["graph"]
    loader: DatasetLoader = SERVICES["loader"]

    if not graph.has_node(kc_id):
        raise HTTPException(status_code=404, detail=f"Knowledge Component {kc_id} not found in graph.")

    # Collect ancestors up to depth
    ancestors = set(graph.predecessors(kc_id))
    two_hop = set()
    for a in ancestors:
        two_hop.update(graph.predecessors(a))
    relevant_nodes = ancestors | two_hop | {kc_id}

    subG = graph.subgraph(relevant_nodes)

    nodes_list = []
    for nid in subG.nodes():
        node_attr = subG.nodes[nid]
        nodes_list.append({
            "id": nid,
            "label": f"KC {nid}",
            "name_en": node_attr.get("name_en", f"Concept_{nid}"),
            "is_target": (nid == kc_id),
            "is_direct": (nid in ancestors),
            "in_degree": subG.in_degree(nid),
            "out_degree": subG.out_degree(nid)
        })

    edges_list = []
    for u, v in subG.edges():
        edges_list.append({
            "source": u,
            "target": v,
            "relation": "prerequisite"
        })

    return {
        "target_kc": kc_id,
        "nodes": nodes_list,
        "edges": edges_list,
        "is_dag": nx.is_directed_acyclic_graph(subG)
    }

def _resolve_mastery_vector(
    student_mastery: Dict[int, float] = None,
    student_history: MasteryEstimateRequest = None
) -> np.ndarray:
    estimator: KnowledgeStateEstimator = SERVICES["estimator"]
    vec = np.full(1175, 0.5, dtype=np.float32)

    if student_history is not None:
        c_arr = np.array(student_history.concepts, dtype=np.int64)
        r_arr = np.array(student_history.responses, dtype=np.int64)
        return estimator.estimate_mastery(c_arr, r_arr)

    if student_mastery is not None:
        for k, v in student_mastery.items():
            if 0 <= int(k) < 1175:
                vec[int(k)] = float(v)
    return vec

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gnn_model_loaded": SERVICES.get("gnn_model") is not None,
        "graph_nodes": SERVICES["graph"].number_of_nodes() if "graph" in SERVICES else 0,
        "graph_edges": SERVICES["graph"].number_of_edges() if "graph" in SERVICES else 0
    }

@app.get("/metadata")
def metadata():
    loader: DatasetLoader = SERVICES["loader"]
    graph: nx.DiGraph = SERVICES["graph"]
    return {
        "dataset": "XES3G5M Mathematics (English Translated)",
        "total_kcs": len(loader.kc_routes_en),
        "total_questions": len(loader.questions_meta),
        "graph_directed_edges": graph.number_of_edges(),
        "graph_is_dag": nx.is_directed_acyclic_graph(graph),
        "api_version": "0.1.0"
    }

@app.get("/concepts/{kc_id}", response_model=ConceptResponse)
def get_concept(kc_id: int):
    graph: nx.DiGraph = SERVICES["graph"]
    loader: DatasetLoader = SERVICES["loader"]
    if not graph.has_node(kc_id):
        raise HTTPException(status_code=404, detail=f"Knowledge Component {kc_id} not found.")

    node = graph.nodes[kc_id]
    preds = list(graph.predecessors(kc_id))
    return ConceptResponse(
        kc_id=kc_id,
        name_en=node.get("name_en", f"Concept_{kc_id}"),
        name_zh=node.get("name_zh"),
        in_degree=graph.in_degree(kc_id),
        out_degree=graph.out_degree(kc_id),
        direct_prerequisites=preds
    )

@app.post("/mastery/estimate", response_model=MasteryEstimateResponse)
def estimate_mastery(req: MasteryEstimateRequest):
    estimator: KnowledgeStateEstimator = SERVICES["estimator"]
    c_arr = np.array(req.concepts, dtype=np.int64)
    r_arr = np.array(req.responses, dtype=np.int64)
    if len(c_arr) != len(r_arr):
        raise HTTPException(status_code=400, detail="Concepts and responses must have identical lengths.")

    m_vec = estimator.estimate_mastery(c_arr, r_arr)
    active_kcs = set(c_arr.tolist())
    profile = {int(c): round(float(m_vec[c]), 4) for c in active_kcs if 0 <= c < 1175}

    return MasteryEstimateResponse(
        num_concepts_evaluated=len(profile),
        active_interactions=len(c_arr),
        overall_mean_mastery=round(float(np.mean(list(profile.values()))), 4) if profile else 0.5,
        mastery_profile=profile
    )

@app.post("/predict", response_model=PredictResponse)
def predict_target(req: PredictRequest):
    cf_engine: CounterfactualImpactEngine = SERVICES["cf_engine"]
    loader: DatasetLoader = SERVICES["loader"]
    if cf_engine is None:
        raise HTTPException(status_code=503, detail="GNN model checkpoint not loaded.")

    m_vec = _resolve_mastery_vector(req.student_mastery, req.student_history)
    with torch.no_grad():
        m_t = torch.tensor(m_vec, dtype=torch.float32, device=cf_engine.device).unsqueeze(0)
        t_t = torch.tensor([req.target_kc], dtype=torch.long, device=cf_engine.device)
        prob = float(cf_engine.model(m_t, t_t, cf_engine.adj_matrix).item())

    return PredictResponse(
        target_kc=req.target_kc,
        target_kc_name_en=loader.get_kc_name(req.target_kc, "en"),
        predicted_success_probability=round(prob, 4),
        model_type="GraphNeuralPredictor (GNN)"
    )

@app.post("/counterfactual/rank", response_model=CounterfactualRankResponse)
def counterfactual_rank(req: CounterfactualRankRequest):
    cf_engine: CounterfactualImpactEngine = SERVICES["cf_engine"]
    loader: DatasetLoader = SERVICES["loader"]
    if cf_engine is None:
        raise HTTPException(status_code=503, detail="GNN model checkpoint not loaded.")

    m_vec = _resolve_mastery_vector(req.student_mastery, req.student_history)
    rankings = cf_engine.compute_counterfactual_impacts(
        target_kc=req.target_kc,
        student_mastery=m_vec,
        candidates=req.candidate_kcs,
        intervention_level=req.intervention_level
    )

    items = []
    factual_pred = 0.5
    if rankings:
        factual_pred = rankings[0]["factual_target_prediction"]
    for r in rankings:
        name_en = loader.get_kc_name(r["kc_id"], "en") if loader is not None else f"Concept_{r['kc_id']}"
        items.append({
            "kc_id": r["kc_id"],
            "kc_name_en": name_en,
            "current_mastery": r["current_mastery"],
            "intervention_level": r.get("actual_intervention_level", r.get("intervention_level", req.intervention_level)),
            "factual_target_prediction": r["factual_target_prediction"],
            "counterfactual_target_prediction": r["counterfactual_target_prediction"],
            "estimated_impact": r["estimated_impact"],
            "relationship": r["relationship"]
        })

    target_name = loader.get_kc_name(req.target_kc, "en") if loader is not None else f"Concept_{req.target_kc}"
    return CounterfactualRankResponse(
        target_kc=req.target_kc,
        target_kc_name_en=target_name,
        intervention_level=req.intervention_level,
        factual_target_prediction=factual_pred,
        rankings=items,
        scientific_disclaimer="Model-based counterfactual estimate under simulated feature intervention; reflects model sensitivity rather than a clinical educational trial result."
    )

@app.post("/learning-path/generate", response_model=LearningPathResponse)
def generate_learning_path(req: LearningPathRequest):
    cf_engine: CounterfactualImpactEngine = SERVICES["cf_engine"]
    path_gen: LearningPathGenerator = SERVICES["path_generator"]
    loader: DatasetLoader = SERVICES["loader"]

    m_vec = _resolve_mastery_vector(req.student_mastery, req.student_history)
    rankings = []
    if cf_engine is not None:
        rankings = cf_engine.compute_counterfactual_impacts(
            target_kc=req.target_kc,
            student_mastery=m_vec,
            intervention_level=req.intervention_level
        )

    path_data = path_gen.generate_path(
        target_kc=req.target_kc,
        student_mastery=m_vec,
        impact_rankings=rankings,
        max_path_length=req.max_path_length
    )

    return LearningPathResponse(
        target_kc=path_data["target_kc"],
        target_kc_name_en=loader.get_kc_name(path_data["target_kc"], "en"),
        total_steps=path_data["total_steps"],
        ordered_kc_sequence=path_data["ordered_kc_sequence"],
        steps=path_data["steps"],
        pedagogical_constraint=path_data["pedagogical_constraint"]
    )
