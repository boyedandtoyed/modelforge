# ModelForge

> Production-grade LLM fine-tuning platform — QLoRA training, real-time experiment tracking, and model serving in a single toolkit.

ModelForge lets you fine-tune any HuggingFace causal language model with 4-bit QLoRA, monitor training in real time via a web dashboard, track experiments locally (or with W&B), and serve the resulting adapter via vLLM — all from one CLI or REST API.

---

## Features

| Feature | Details |
|---------|---------|
| **QLoRA fine-tuning** | 4-bit NF4 quantization (BitsAndBytes) + LoRA adapters (PEFT) + SFT (TRL) |
| **Live dashboard** | Single-page app: loss curve chart, progress bars, per-run config viewer |
| **SSE streaming** | Training progress pushed to UI via Server-Sent Events — no polling |
| **Experiment tracking** | Local JSON logs (W&B-compatible interface); plug in your W&B key for cloud sync |
| **REST API** | FastAPI with auto-generated Swagger docs at `/docs` |
| **Model serving** | Demo echo endpoint; production path uses vLLM with LoRA adapter |
| **CLI** | `modelforge train / runs / serve` backed by Typer + Rich |
| **Evaluator** | Perplexity and BLEU-1/2 metric computation |
| **Docker** | Single-image deploy with `docker compose up` |

---

## Quick Start (Demo Mode)

```bash
git clone <repo-url> && cd modelforge
pip install -e .

# Fine-tune with demo simulation (no GPU needed)
modelforge train --model meta-llama/Llama-2-7b-hf --dataset tatsu-lab/alpaca --epochs 3

# List all experiment runs
modelforge runs

# Start the web dashboard + REST API
modelforge serve --port 8080
# Open http://localhost:8080
```

---

## Web Dashboard

Start the server and open `http://localhost:8080`:

- **Dashboard** — stats cards, recent runs table, CLI quick-reference
- **Train** — config form, live progress bar, real-time loss curve chart
- **Experiment Runs** — full run history, expandable detail rows with per-run charts
- **Playground** — prompt your fine-tuned model (demo or live vLLM)

---

## Installation

### Minimal (CLI + API + Dashboard)

```bash
pip install -e .
```

### Full (GPU Training)

```bash
pip install -e .
pip install torch transformers peft trl bitsandbytes wandb vllm
```

### Development

```bash
pip install -e ".[dev]"
python3 -m pytest          # 31 tests, ~0.5s
```

---

## CLI Reference

```bash
# Fine-tune
modelforge train \
  --model  meta-llama/Llama-2-7b-hf \
  --dataset tatsu-lab/alpaca \
  --epochs 3 \
  --samples 1000 \
  --output ./outputs

# Load config from JSON
modelforge train --config my_config.json

# List runs
modelforge runs

# Serve (demo mode by default; use vLLM adapter for production)
modelforge serve --adapter ./outputs/adapter --port 8080
```

---

## REST API

The server exposes a versioned REST API under `/api`. Swagger docs available at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/train` | Start a training run |
| `GET` | `/api/train/status` | Check if training is active |
| `GET` | `/api/train/stream` | SSE stream of training events |
| `GET` | `/api/runs` | List all experiment runs |
| `GET` | `/api/runs/{run_id}` | Get a specific run |
| `DELETE` | `/api/runs/{run_id}` | Delete a run |
| `GET` | `/api/stats` | Dashboard summary stats |
| `POST` | `/api/generate` | Run inference on the served model |
| `GET` | `/api/health` | Health check |

### Example: start training via API

```bash
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "meta-llama/Llama-2-7b-hf",
    "dataset": "tatsu-lab/alpaca",
    "num_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 32
  }'
```

---

## Configuration

Training is configured via `TrainingConfig` (Pydantic v2). All fields have sensible defaults.

```python
from modelforge import TrainingConfig, LoRAConfig

cfg = TrainingConfig(
    base_model="mistralai/Mistral-7B-v0.1",
    dataset="databricks/databricks-dolly-15k",
    num_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lora=LoRAConfig(r=64, lora_alpha=128),
    report_to="wandb",
    experiment_name="mistral-dolly-run1",
)
```

Or from a JSON config file:

```json
{
  "base_model": "meta-llama/Llama-2-7b-hf",
  "dataset": "tatsu-lab/alpaca",
  "num_epochs": 3,
  "lora": { "r": 16, "lora_alpha": 32 }
}
```

---

## Architecture

```
modelforge/
├── config.py      Pydantic models: TrainingConfig, LoRAConfig, QuantizationConfig
├── data.py        HuggingFace dataset loading + Alpaca/Chat prompt templating
├── trainer.py     QLoRA training loop (demo sim + GPU path stubs)
├── tracker.py     Experiment tracking → ./logs/<run_id>.json
├── evaluator.py   Perplexity + BLEU metrics
├── utils.py       Logging, seeding, formatting helpers
├── api.py         FastAPI router: REST endpoints + SSE training stream
├── serve.py       FastAPI app: mounts router, serves SPA
├── cli.py         Typer CLI: train / runs / serve
└── ui/
    └── index.html Single-page dashboard (Alpine.js + Chart.js + Tailwind CDN)
```

### Data Flow

```
CLI: modelforge train
  └─► TrainingConfig → load_dataset → trainer.train → ExperimentTracker.log_run

Web UI → POST /api/train
  └─► Background thread: load_dataset → trainer.train (progress_cb → Queue)
      GET /api/train/stream (SSE) → EventSource in browser → live Chart.js
```

---

## Production Upgrade

ModelForge runs fully in demo mode out of the box (no GPU, no model downloads). To enable real training, install the GPU stack and set environment variables:

```bash
cp .env.example .env
# Add HF_TOKEN and WANDB_API_KEY
pip install torch transformers peft trl bitsandbytes wandb vllm
```

The GPU code paths are documented as inline comments in [trainer.py](src/modelforge/trainer.py) — each stub shows the exact production imports and implementation.

---

## Docker

```bash
cp .env.example .env   # configure HF_TOKEN etc.
docker compose up
# Dashboard: http://localhost:8080
```

---

## Testing

```bash
python3 -m pytest                    # all 31 tests
python3 -m pytest tests/test_api.py  # API tests only
python3 -m pytest --cov=modelforge   # with coverage report
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Fine-tuning | QLoRA — PEFT + BitsAndBytes 4-bit |
| Training loop | TRL SFTTrainer |
| Datasets | HuggingFace `datasets` |
| Experiment tracking | Local JSON / Weights & Biases |
| Model serving | vLLM (production) |
| CLI | Typer + Rich |
| API | FastAPI + Pydantic v2 + SSE |
| Dashboard | Alpine.js + Chart.js + Tailwind CSS |
| Tests | pytest + httpx |
| Container | Docker + Compose |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Author

Binod Tiwari
