"""Tests for ExperimentTracker."""
from __future__ import annotations

from modelforge.config import TrainingConfig
from modelforge.trainer import train
from modelforge.tracker import ExperimentTracker


def test_log_run_creates_file(tmp_tracker: ExperimentTracker, default_config: TrainingConfig, tiny_samples):
    from modelforge.data import _synthetic_samples
    run = train(default_config, tiny_samples)
    path = tmp_tracker.log_run(run)
    import pathlib
    assert pathlib.Path(path).exists()


def test_load_runs_returns_list(tmp_tracker: ExperimentTracker, default_config: TrainingConfig, tiny_samples):
    run = train(default_config, tiny_samples)
    tmp_tracker.log_run(run)

    runs = tmp_tracker.load_runs()
    assert len(runs) == 1
    r = runs[0]
    assert "run_id" in r
    assert "config" in r
    assert "metrics" in r
    assert "best_loss" in r
    assert r["best_loss"] < 2.6


def test_load_runs_empty_dir(tmp_tracker: ExperimentTracker):
    runs = tmp_tracker.load_runs()
    assert runs == []


def test_run_id_contains_experiment_name(tmp_tracker: ExperimentTracker, default_config: TrainingConfig, tiny_samples):
    run = train(default_config, tiny_samples)
    tmp_tracker.log_run(run)

    runs = tmp_tracker.load_runs()
    assert runs[0]["run_id"].startswith("test-experiment")


def test_multiple_runs_sorted_newest_first(tmp_tracker: ExperimentTracker, default_config: TrainingConfig, tiny_samples):
    for _ in range(3):
        tracker = ExperimentTracker("test-experiment", log_dir=tmp_tracker.log_dir)
        run = train(default_config, tiny_samples)
        tracker.log_run(run)

    runs = tmp_tracker.load_runs()
    assert len(runs) == 3
    ids = [r["run_id"] for r in runs]
    assert ids == sorted(ids, reverse=True)
