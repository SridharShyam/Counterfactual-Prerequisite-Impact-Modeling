# Counterfactual Prerequisite Impact Modeling (C-PrIM)

A research framework for personalized learning path recommendation using knowledge graphs, graph neural networks, and counterfactual impact estimation.

Rather than relying on heuristics alone, C-PrIM estimates how improving specific prerequisite concepts influences a student's probability of mastering a target concept, then generates a prerequisite-compliant study sequence.

---

## Key Highlights

- **Prerequisite Graph Modeling**: Builds directed acyclic graphs (DAGs) from educational curriculum structures and student interaction transitions across 1,175 knowledge components (KCs).
- **Graph Neural Prediction**: Uses graph propagation over normalized adjacency matrices with residual connections to model prerequisite dependencies.
- **Counterfactual Impact Estimation**: Estimates the expected improvement on a target concept under simulated prerequisite mastery interventions:
  $$\Delta I_i = \hat{P}(Y_T = 1 \mid \text{do}(M_i = \theta)) - \hat{P}(Y_T = 1 \mid M)$$
- **Constrained Path Generation**: Produces personalized study sequences that respect topological prerequisite order while prioritizing high-impact concepts.
- **Interactive Research Dashboard**: Includes a FastAPI backend and lightweight web interface for testing interventions and inspecting learning paths.

---

## Repository Structure

```text
CFLP_Project/
├── configs/
│   └── config.yaml             # Dataset paths and model hyperparameters
├── notebooks/                  # Interactive exploratory analysis notebooks
│   ├── 01_dataset_exploration.ipynb
│   └── 02_interaction_insights.ipynb
├── src/
│   └── cflp/
│       ├── api/                # FastAPI service, endpoints, and dashboard
│       ├── counterfactual/     # Counterfactual intervention engine
│       ├── data/               # Sequence parsing and XES3G5M data loaders
│       ├── evaluation/         # Metrics (ROC-AUC, PR-AUC, Brier score, etc.)
│       ├── graph/              # Prerequisite DAG builder and topological tools
│       ├── models/             # Baseline logistic and GNN predictors
│       └── paths/              # Pedagogical learning path generator
├── scripts/
│   ├── audit_dataset.py        # Inspects dataset distributions and splits
│   ├── build_graph.py          # Builds and validates the prerequisite graph
│   ├── run_experiment.py       # Runs single-run counterfactual experiments
│   ├── run_reproducible_pipeline.py  # End-to-end multi-seed benchmark and figure generator
│   └── train_models.py         # Trains baseline and GNN models
├── tests/                      # Automated test suite (11 test cases)
├── pyproject.toml              # Project metadata and test configuration
└── requirements.txt            # Python dependencies
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/SridharShyam/Counterfactual-Prerequisite-Impact-Modeling.git
cd Counterfactual-Prerequisite-Impact-Modeling/CFLP_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests

```bash
# On Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:PYTHONPATH="src;."
pytest -v
```

### 3. Launch the Research Dashboard

```bash
uvicorn cflp.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` in your browser to inspect prerequisite graphs, simulate mastery interventions, and generate personalized learning paths. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 4. Reproduce Benchmark Experiments

```bash
# Run full training and generate all evaluation figures/tables
python scripts/run_reproducible_pipeline.py
```

---

## Research Disclaimer

Counterfactual impact values in this repository are model-based estimates derived from simulated feature interventions over observed curriculum taxonomies. They reflect model sensitivity rather than verified causal effects from randomized educational trials.
