"""Integration tests for the FastAPI REST API endpoints."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient


def test_health(test_client: TestClient):
    r = test_client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_stats_structure(test_client: TestClient):
    r = test_client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_runs" in data
    assert "best_loss" in data
    assert "total_hours" in data
    assert "training_active" in data


def test_runs_empty(test_client: TestClient):
    r = test_client.get("/api/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_generate_returns_text(test_client: TestClient):
    r = test_client.post("/api/generate", json={"prompt": "What is a transformer?"})
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    assert "prompt_tokens" in data
    assert "completion_tokens" in data
    assert "finish_reason" in data


def test_generate_prompt_reflected(test_client: TestClient):
    prompt = "Hello ModelForge"
    r = test_client.post("/api/generate", json={"prompt": prompt})
    assert prompt[:20] in r.json()["text"]


def test_generate_invalid_temperature(test_client: TestClient):
    r = test_client.post("/api/generate", json={"prompt": "x", "temperature": 5.0})
    assert r.status_code == 422


def test_get_run_not_found(test_client: TestClient):
    r = test_client.get("/api/runs/nonexistent-run-id")
    assert r.status_code == 404


def test_delete_run_not_found(test_client: TestClient):
    r = test_client.delete("/api/runs/nonexistent-run-id")
    assert r.status_code == 404


def test_training_status(test_client: TestClient):
    r = test_client.get("/api/train/status")
    assert r.status_code == 200
    assert "active" in r.json()


def test_train_start_returns_started(test_client: TestClient):
    import modelforge.api as api_module
    api_module._train_active = False

    r = test_client.post("/api/train", json={
        "base_model": "meta-llama/Llama-2-7b-hf",
        "dataset": "tatsu-lab/alpaca",
        "num_epochs": 1,
        "max_samples": 10,
    })
    assert r.status_code == 200
    assert r.json()["status"] == "started"

    # give the background thread a moment, then reset flag
    time.sleep(0.1)
    api_module._train_active = False


def test_train_conflict(test_client: TestClient):
    import modelforge.api as api_module
    api_module._train_active = True

    try:
        r = test_client.post("/api/train", json={
            "base_model": "meta-llama/Llama-2-7b-hf",
            "dataset": "tatsu-lab/alpaca",
            "num_epochs": 1,
            "max_samples": 10,
        })
        assert r.status_code == 409
    finally:
        api_module._train_active = False


def test_root_serves_html(test_client: TestClient):
    r = test_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
