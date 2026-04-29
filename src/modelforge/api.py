"""REST API endpoints for the ModelForge UI dashboard."""
from __future__ import annotations
import json
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import TrainingConfig, LoRAConfig
from .tracker import ExperimentTracker
from .data import load_dataset

router = APIRouter(prefix="/api")

_train_queue: Queue = Queue()
_train_active: bool = False


class TrainRequest(BaseModel):
    base_model: str = "meta-llama/Llama-2-7b-hf"
    dataset: str = "tatsu-lab/alpaca"
    num_epochs: int = Field(3, ge=1, le=20)
    max_samples: int = Field(200, ge=10, le=10000)
    learning_rate: float = Field(2e-4, gt=0)
    per_device_train_batch_size: int = Field(4, ge=1, le=64)
    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1)
    output_dir: str = "./outputs"
    experiment_name: str = "modelforge-run"


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.01, le=1.0)


def _training_worker(cfg: TrainingConfig) -> None:
    global _train_active
    try:
        samples = load_dataset(cfg.dataset, cfg.dataset_split, cfg.max_samples)
        _train_queue.put({"type": "loaded", "data": {"samples": len(samples)}})

        from .trainer import train as run_train

        tracker = ExperimentTracker(cfg.experiment_name)

        def progress_cb(metrics, total_steps):
            _train_queue.put({
                "type": "progress",
                "data": {
                    "step": metrics.step,
                    "total": total_steps,
                    "loss": metrics.loss,
                    "lr": metrics.learning_rate,
                    "epoch": metrics.epoch,
                    "tokens_per_sec": metrics.tokens_per_second,
                    "grad_norm": metrics.grad_norm,
                },
            })
            time.sleep(0.025)

        run = run_train(cfg, samples, progress_cb)
        log_path = tracker.log_run(run)

        _train_queue.put({
            "type": "complete",
            "data": {
                "best_loss": round(run.best_loss, 4),
                "duration": round((run.end_time or time.time()) - run.start_time, 1),
                "steps": len(run.metrics),
                "run_id": tracker._run_id,
                "log_path": log_path,
            },
        })
    except Exception as exc:
        _train_queue.put({"type": "error", "data": {"message": str(exc)}})
    finally:
        _train_active = False


@router.post("/train")
def start_training(req: TrainRequest):
    global _train_active, _train_queue
    if _train_active:
        raise HTTPException(status_code=409, detail="Training already in progress")

    _train_active = True
    while not _train_queue.empty():
        _train_queue.get_nowait()

    cfg = TrainingConfig(
        base_model=req.base_model,
        dataset=req.dataset,
        num_epochs=req.num_epochs,
        max_samples=req.max_samples,
        learning_rate=req.learning_rate,
        per_device_train_batch_size=req.per_device_train_batch_size,
        lora=LoRAConfig(r=req.lora_r, lora_alpha=req.lora_alpha),
        output_dir=req.output_dir,
        experiment_name=req.experiment_name,
    )

    threading.Thread(target=_training_worker, args=(cfg,), daemon=True).start()
    return {"status": "started"}


@router.get("/train/status")
def training_status():
    return {"active": _train_active}


@router.get("/train/stream")
async def train_stream():
    import asyncio

    async def event_generator():
        while True:
            try:
                event = _train_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("complete", "error"):
                    break
            except Empty:
                if not _train_active:
                    break
                await asyncio.sleep(0.05)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs")
def list_runs():
    tracker = ExperimentTracker("modelforge")
    return tracker.load_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    for run in ExperimentTracker("modelforge").load_runs():
        if run.get("run_id") == run_id:
            return run
    raise HTTPException(status_code=404, detail="Run not found")


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    for f in Path("./logs").glob(f"{run_id}.json"):
        f.unlink()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Run not found")


@router.get("/stats")
def get_stats():
    all_runs = ExperimentTracker("modelforge").load_runs()
    best_loss = min((r.get("best_loss", float("inf")) for r in all_runs), default=None)
    total_duration = sum(r.get("duration_seconds", 0) for r in all_runs)
    return {
        "total_runs": len(all_runs),
        "best_loss": round(best_loss, 4) if best_loss and best_loss != float("inf") else None,
        "total_hours": round(total_duration / 3600, 3),
        "training_active": _train_active,
    }


@router.post("/generate")
def generate(req: GenerateRequest):
    demo_text = (
        f'[ModelForge Demo] You asked: "{req.prompt[:100]}{"..." if len(req.prompt) > 100 else ""}". '
        "In production this endpoint connects to a vLLM-served fine-tuned adapter. "
        "Run: modelforge serve --adapter ./outputs/adapter"
    )
    return {
        "text": demo_text,
        "prompt_tokens": len(req.prompt.split()),
        "completion_tokens": len(demo_text.split()),
        "finish_reason": "stop",
        "model": "modelforge-demo",
    }


@router.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0", "mode": "demo", "training_active": _train_active}
