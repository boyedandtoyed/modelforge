"""Shared pytest fixtures for the ModelForge test suite."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelforge.config import TrainingConfig
from modelforge.data import _synthetic_samples
from modelforge.tracker import ExperimentTracker


@pytest.fixture()
def default_config() -> TrainingConfig:
    """Minimal TrainingConfig: 1 epoch, 10 samples, batch 2."""
    return TrainingConfig(
        num_epochs=1,
        max_samples=10,
        per_device_train_batch_size=2,
        experiment_name="test-run",
    )


@pytest.fixture()
def tiny_samples():
    """10 synthetic Sample objects."""
    return _synthetic_samples(10)


@pytest.fixture()
def test_client():
    """FastAPI TestClient bound to the main serve app."""
    from modelforge.serve import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture()
def tmp_tracker(tmp_path: Path) -> ExperimentTracker:
    """ExperimentTracker writing to a temporary directory."""
    return ExperimentTracker("test-experiment", log_dir=str(tmp_path))
