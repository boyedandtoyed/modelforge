"""Experiment tracking (W&B-compatible interface)."""
from __future__ import annotations
import json
import time
from pathlib import Path
from .trainer import TrainingRun


class ExperimentTracker:
    def __init__(self, experiment_name: str, log_dir: str = "./logs") -> None:
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = f"{experiment_name}-{int(time.time())}"

    def log_run(self, run: TrainingRun) -> str:
        data = {
            "run_id": self._run_id,
            "config": run.config.model_dump(),
            "metrics": [
                {
                    "epoch": m.epoch,
                    "step": m.step,
                    "loss": m.loss,
                    "lr": m.learning_rate,
                    "tokens_per_sec": m.tokens_per_second,
                }
                for m in run.metrics
            ],
            "best_loss": run.best_loss,
            "duration_seconds": (run.end_time or time.time()) - run.start_time,
        }
        path = self.log_dir / f"{self._run_id}.json"
        path.write_text(json.dumps(data, indent=2))
        return str(path)

    def load_runs(self) -> list[dict]:
        runs = []
        for f in self.log_dir.glob("*.json"):
            try:
                runs.append(json.loads(f.read_text()))
            except Exception:
                pass
        return sorted(runs, key=lambda r: r.get("run_id", ""), reverse=True)
