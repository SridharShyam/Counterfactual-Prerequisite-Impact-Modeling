import os
import json
from fastapi.testclient import TestClient
from cflp.api.main import app

def test_api_workflow():
    client = TestClient(app)
    client.__enter__()

    # 1. Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["gnn_model_loaded"] is True
    print("Health check: PASS")

    # 2. Metadata
    r = client.get("/metadata")
    assert r.status_code == 200
    assert r.json()["total_kcs"] == 1175
    print("Metadata check: PASS")

    # 3. Concept details
    r = client.get("/concepts/4")
    assert r.status_code == 200
    assert r.json()["kc_id"] == 4
    print("Concept check: PASS")

    # 4. Mastery estimation
    payload_m = {"concepts": [0, 2, 2, 3], "responses": [1, 0, 1, 1]}
    r = client.post("/mastery/estimate", json=payload_m)
    assert r.status_code == 200
    assert "mastery_profile" in r.json()
    print("Mastery estimation check: PASS")

    # 5. Target prediction
    payload_pred = {"target_kc": 4, "student_history": payload_m}
    r = client.post("/predict", json=payload_pred)
    assert r.status_code == 200
    assert 0.0 <= r.json()["predicted_success_probability"] <= 1.0
    print("Predict target check: PASS")

    # 6. Counterfactual impact ranking
    payload_cf = {"target_kc": 4, "student_history": payload_m, "intervention_level": 0.85}
    r = client.post("/counterfactual/rank", json=payload_cf)
    assert r.status_code == 200
    rankings = r.json().get("rankings", [])
    assert len(rankings) > 0
    print(f"Counterfactual rank check: PASS ({len(rankings)} candidates ranked)")

    # 7. Learning path generation
    r = client.post("/learning-path/generate", json=payload_cf)
    assert r.status_code == 200
    steps = r.json().get("steps", [])
    assert len(steps) > 0
    print(f"Learning path generation check: PASS ({len(steps)} steps generated)")

if __name__ == "__main__":
    test_api_workflow()
