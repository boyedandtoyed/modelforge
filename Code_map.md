# ModelForge — Code Map

> Read this file first to restore context in any new session.
> Updated: 2026-05-10

---

## Project Overview

ModelForge is a production-grade LLM fine-tuning platform. It provides:
- **CLI** (`modelforge train/runs/serve`) backed by Typer + Rich
- **REST API** (FastAPI + SSE streaming) for programmatic access
- **Web Dashboard** (single-page app; Alpine.js + Chart.js + Tailwind)
- **QLoRA pipeline** (simulated with realistic curves; GPU path is documented inline)
- **Experiment tracking** (local JSON; W&B-compatible interface)
- **Model serving** (demo echo; production uses vLLM)

---

## File / Folder Structure

```
modelforge/
├── Code_map.md              ← This file
├── Dockerfile               ← Single-stage Python 3.11 slim image
├── docker-compose.yml       ← Local dev/demo compose stack
├── .env.example             ← Environment variable template
├── .gitignore               ← Python/IDE/venv ignores
├── pyproject.toml           ← Build config, deps, pytest settings
├── README.md                ← Project overview & quick-start
├── CONTRIBUTING.md          ← Dev setup, coding conventions, PR guide
│
├── src/
│   └── modelforge/
│       ├── __init__.py      ← Public API surface & __version__
│       ├── config.py        ← Pydantic config models (LoRA, Quantization, Training)
│       ├── data.py          ← Dataset loading, prompt formatting, Sample dataclass
│       ├── trainer.py       ← QLoRA training loop (simulated + GPU-path stubs)
│       ├── tracker.py       ← Experiment tracking, JSON persistence
│       ├── evaluator.py     ← Perplexity, BLEU, metric aggregation
│       ├── utils.py         ← Logging setup, seed, formatting helpers
│       ├── api.py           ← FastAPI router: /api/* endpoints + SSE stream
│       ├── serve.py         ← FastAPI app: mounts router, serves SPA
│       └── ui/
│           └── index.html   ← Single-page dashboard (Alpine.js + Chart.js)
│
└── tests/
    ├── __init__.py
    ├── conftest.py          ← Shared pytest fixtures
    ├── test_config.py       ← Config model validation tests
    ├── test_data.py         ← Dataset loading and formatting tests
    ├── test_trainer.py      ← Training loop tests
    ├── test_tracker.py      ← Experiment tracker tests
    ├── test_evaluator.py    ← Evaluator metric tests
    └── test_api.py          ← FastAPI endpoint integration tests
```

---

## Module Details

### `src/modelforge/__init__.py`
**Purpose:** Public package surface. Exposes version and main entry-points so downstream code can do `from modelforge import TrainingConfig`.

| Symbol | Type | Description |
|--------|------|-------------|
| `__version__` | `str` | Semantic version, e.g. `"0.1.0"` |
| `TrainingConfig` | class | Re-export from `config` |
| `LoRAConfig` | class | Re-export from `config` |
| `QuantizationConfig` | class | Re-export from `config` |
| `train` | function | Re-export from `trainer` |
| `load_dataset` | function | Re-export from `data` |
| `ExperimentTracker` | class | Re-export from `tracker` |
| `Evaluator` | class | Re-export from `evaluator` |

**Status:** ✅ Complete

---

### `src/modelforge/config.py`
**Purpose:** Strongly-typed training configuration using Pydantic v2. Central source of truth for all hyperparameters.

**Imports:** `pydantic.BaseModel`, `pydantic.Field`

| Class | Key Fields | Description |
|-------|-----------|-------------|
| `LoRAConfig` | `r`, `lora_alpha`, `target_modules`, `lora_dropout`, `bias`, `task_type` | LoRA adapter hyperparameters |
| `QuantizationConfig` | `load_in_4bit`, `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant` | BitsAndBytes 4-bit quantization settings |
| `TrainingConfig` | `base_model`, `output_dir`, `dataset`, `num_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `lr_scheduler_type`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `lora`, `quantization`, `experiment_name`, `report_to`, `serve_port`, `max_new_tokens`, `temperature` | Full training configuration, composes LoRAConfig and QuantizationConfig |

**Status:** ✅ Complete

---

### `src/modelforge/data.py`
**Purpose:** Dataset loading, prompt templating, and preprocessing for instruction fine-tuning.

**Imports:** `dataclasses`, `typing`, `datasets` (optional — graceful fallback)

| Symbol | Type | Signature | Description |
|--------|------|-----------|-------------|
| `ALPACA_TEMPLATE` | `str` | — | Alpaca-style `### Instruction / Input / Response` template |
| `CHAT_TEMPLATE` | `str` | — | ChatML-style `<|system|>/<|user|>/<|assistant|>` template |
| `Sample` | `@dataclass` | `prompt: str, completion: str, tokens: int = 0` | Single training example |
| `format_alpaca` | `fn` | `(row: dict) → Sample` | Formats an Alpaca-style row into a Sample |
| `format_chat` | `fn` | `(row: dict) → Sample` | Formats a chat-style row into a Sample |
| `load_dataset` | `fn` | `(dataset_id: str, split: str, max_samples: int|None) → list[Sample]` | Loads HuggingFace dataset, falls back to synthetic data |
| `_synthetic_samples` | `fn` | `(n: int) → list[Sample]` | Returns n synthetic Alpaca-formatted training samples |

**Status:** ✅ Complete

---

### `src/modelforge/trainer.py`
**Purpose:** QLoRA fine-tuning pipeline. Runs a simulated training loop with realistic loss curves in demo mode; GPU path is documented inline with exact production imports.

**Imports:** `time`, `math`, `dataclasses`, `.config`, `.data`

| Symbol | Type | Signature | Description |
|--------|------|-----------|-------------|
| `TrainingMetrics` | `@dataclass` | `epoch, step, loss, learning_rate, tokens_per_second, grad_norm` | Per-step training metrics |
| `TrainingRun` | `@dataclass` | `config, metrics, start_time, end_time, best_loss, adapter_path` | Full run record |
| `train` | `fn` | `(config: TrainingConfig, samples: list[Sample], progress_cb=None) → TrainingRun` | Main training entry-point; simulates QLoRA loop, calls `progress_cb` each step |
| `merge_adapter` | `fn` | `(base_model: str, adapter_path: str, output_path: str) → str` | Merges LoRA adapter into base model (production: `peft.PeftModel.merge_and_unload`) |
| `_build_qlora_model` | `fn` | `(cfg: TrainingConfig) → tuple[model, tokenizer]` | GPU path: loads model with BitsAndBytes 4-bit quant + LoRA config |
| `_get_sft_trainer` | `fn` | `(model, tokenizer, cfg, samples) → SFTTrainer` | GPU path: constructs TRL SFTTrainer with cosine schedule |

**Status:** ✅ Complete (simulated loop + documented GPU stubs)

---

### `src/modelforge/tracker.py`
**Purpose:** W&B-compatible experiment tracking backed by local JSON files. Each run persists to `./logs/<run-id>.json`.

**Imports:** `json`, `time`, `pathlib`, `.trainer`

| Symbol | Type | Signature | Description |
|--------|------|-----------|-------------|
| `ExperimentTracker` | class | `__init__(experiment_name: str, log_dir: str = "./logs")` | Tracker instance for one experiment |
| `ExperimentTracker.log_run` | method | `(run: TrainingRun) → str` | Serializes run to JSON, returns file path |
| `ExperimentTracker.load_runs` | method | `() → list[dict]` | Loads all runs from log_dir, sorted newest-first |

**Status:** ✅ Complete

---

### `src/modelforge/evaluator.py`
**Purpose:** Post-training evaluation — perplexity computation, n-gram metrics (BLEU/ROUGE proxies), and run-level metric aggregation.

**Imports:** `math`, `dataclasses`, `collections`, `.data`

| Symbol | Type | Signature | Description |
|--------|------|-----------|-------------|
| `EvalResult` | `@dataclass` | `perplexity, avg_loss, bleu1, bleu2, num_samples, eval_time` | Aggregated evaluation output |
| `Evaluator` | class | `__init__(model_path: str = "demo")` | Evaluation engine |
| `Evaluator.compute_perplexity` | method | `(samples: list[Sample]) → float` | Computes mean token-level cross-entropy perplexity (demo: analytic formula) |
| `Evaluator.compute_bleu` | method | `(samples: list[Sample], n: int = 1) → float` | Computes n-gram precision (BLEU-1 and BLEU-2) |
| `Evaluator.evaluate` | method | `(samples: list[Sample]) → EvalResult` | Full evaluation pass: perplexity + BLEU |
| `_ngrams` | `fn` | `(tokens: list[str], n: int) → Counter` | Returns Counter of n-gram occurrences |

**Status:** ✅ Complete

---

### `src/modelforge/utils.py`
**Purpose:** Shared utilities used across the package — structured logging, random seed, number formatting.

**Imports:** `logging`, `random`, `os`

| Symbol | Type | Signature | Description |
|--------|------|-----------|-------------|
| `get_logger` | `fn` | `(name: str) → logging.Logger` | Returns a configured logger with rich-compatible formatting |
| `set_seed` | `fn` | `(seed: int) → None` | Sets Python, NumPy, and (optionally) PyTorch random seeds for reproducibility |
| `fmt_loss` | `fn` | `(loss: float) → str` | Formats loss to 4 decimal places |
| `fmt_duration` | `fn` | `(seconds: float) → str` | Human-readable duration: `"1m 23s"` |
| `human_size` | `fn` | `(num_params: int) → str` | Formats parameter count: `"7.0 B"`, `"560 M"` |

**Status:** ✅ Complete

---

### `src/modelforge/api.py`
**Purpose:** FastAPI `APIRouter` exposing all REST endpoints consumed by the dashboard and CLI integrations.

**Imports:** `json`, `time`, `threading`, `queue`, `pathlib`, `fastapi`, `pydantic`, `.config`, `.tracker`, `.data`

| Symbol | Type | Route | Description |
|--------|------|-------|-------------|
| `TrainRequest` | Pydantic model | — | Request body for POST /api/train |
| `GenerateRequest` | Pydantic model | — | Request body for POST /api/generate |
| `_training_worker` | `fn` | — | Background thread: runs training, pushes events to `_train_queue` |
| `start_training` | endpoint | `POST /api/train` | Starts background training, returns `{"status":"started"}` |
| `training_status` | endpoint | `GET /api/train/status` | Returns `{"active": bool}` |
| `train_stream` | endpoint | `GET /api/train/stream` | SSE stream of `progress`/`complete`/`error` events |
| `list_runs` | endpoint | `GET /api/runs` | Returns all experiment runs (sorted newest-first) |
| `get_run` | endpoint | `GET /api/runs/{run_id}` | Returns single run or 404 |
| `delete_run` | endpoint | `DELETE /api/runs/{run_id}` | Deletes run JSON file |
| `get_stats` | endpoint | `GET /api/stats` | Dashboard summary: total_runs, best_loss, total_hours, training_active |
| `generate` | endpoint | `POST /api/generate` | Inference (demo echo; production: vLLM adapter) |
| `health` | endpoint | `GET /api/health` | Health check: version, mode, training_active |

**Module-level state:**
- `_train_queue: Queue` — inter-thread event channel
- `_train_active: bool` — mutex flag preventing concurrent runs

**Status:** ✅ Complete

---

### `src/modelforge/serve.py`
**Purpose:** Root FastAPI application — mounts the API router and serves the SPA index.html at `/`.

**Imports:** `pathlib`, `fastapi`, `fastapi.responses`, `.api`

| Symbol | Type | Route | Description |
|--------|------|-------|-------------|
| `app` | `FastAPI` | — | Application instance (docs at `/docs`, redoc at `/redoc`) |
| `index` | endpoint | `GET /` | Serves `ui/index.html` |

**Status:** ✅ Complete

---

### `src/modelforge/cli.py`
**Purpose:** Typer CLI application with three commands, rendered with Rich progress bars and tables.

**Imports:** `json`, `pathlib`, `typer`, `rich`, `.config`, `.data`, `.trainer`, `.tracker`, `.serve`

| Symbol | Type | CLI signature | Description |
|--------|------|--------------|-------------|
| `app` | `typer.Typer` | — | Root CLI application |
| `train` | command | `modelforge train [--config FILE] [--model STR] [--dataset STR] [--epochs INT] [--samples INT] [--output STR]` | Fine-tune with QLoRA; shows Rich progress bar |
| `runs` | command | `modelforge runs` | Lists all experiment runs in a Rich table |
| `serve` | command | `modelforge serve [--adapter STR] [--port INT]` | Starts uvicorn on the FastAPI app |

**Status:** ✅ Complete

---

### `src/modelforge/ui/index.html`
**Purpose:** Self-contained single-page application dashboard. Zero build step — loads dependencies from CDN.

**Dependencies (CDN):** Tailwind CSS v3, Chart.js v4.4.3, Alpine.js v3

**Pages / sections:**

| Page | Alpine component key | Features |
|------|---------------------|----------|
| Dashboard | `page === 'dashboard'` | Stat cards (runs, best loss, compute hours, status), recent experiments table, CLI quick reference |
| Train Model | `page === 'train'` | Config form (model, dataset, epochs, LoRA), launch button, live SSE progress bar, real-time loss chart |
| Experiment Runs | `page === 'runs'` | Sortable table, expandable rows with per-run Chart.js loss curve, delete action |
| Playground | `page === 'playground'` | Prompt textarea, temperature/top-p/max-tokens sliders, generate button, response display |

**Alpine.js `app()` methods:**

| Method | Description |
|--------|-------------|
| `init()` | Loads stats + runs, initializes Chart.js canvas |
| `loadStats()` | `GET /api/stats` → `this.stats` |
| `loadRuns()` | `GET /api/runs` → `this.runs` |
| `startTraining()` | `POST /api/train` then opens EventSource for SSE stream |
| `initLiveChart()` | Creates/resets Chart.js line chart on `#liveChart` |
| `pushLiveChart(step, loss)` | Appends data point, trims to 300 max |
| `toggleRun(run)` | Expands/collapses run row, renders per-run chart |
| `initRunChart(run)` | Creates Chart.js chart from stored metrics |
| `deleteRun(runId)` | `DELETE /api/runs/{id}`, refreshes table |
| `generate()` | `POST /api/generate` → `this.genResult` |
| `progressPct` | Computed: `step / total * 100` |
| `fmtDur(secs)` | Duration formatter (`"1m 23s"`) |

**Status:** ✅ Complete

---

## Test Suite

### `tests/conftest.py`
**Purpose:** Shared pytest fixtures.

| Fixture | Scope | Description |
|---------|-------|-------------|
| `default_config` | function | `TrainingConfig` with minimal settings (1 epoch, 10 samples) |
| `tiny_samples` | function | 10 synthetic `Sample` objects |
| `test_client` | function | FastAPI `TestClient` for API tests |
| `tmp_tracker` | function | `ExperimentTracker` with a temp directory |

**Status:** ✅ Complete

---

### `tests/test_config.py`
| Test | Description |
|------|-------------|
| `test_default_config` | Default values are correct |
| `test_custom_lora` | Custom LoRA rank/alpha accepted |

**Status:** ✅ Complete

---

### `tests/test_data.py`
| Test | Description |
|------|-------------|
| `test_format_alpaca` | Instruction appears in prompt, output maps to completion |
| `test_synthetic_samples` | Correct count, non-empty prompts |

**Status:** ✅ Complete

---

### `tests/test_trainer.py`
| Test | Description |
|------|-------------|
| `test_train_completes` | Run finishes, best_loss < 2.6, metrics populated |

**Status:** ✅ Complete

---

### `tests/test_tracker.py`
| Test | Description |
|------|-------------|
| `test_log_run_creates_file` | JSON file written to log_dir |
| `test_load_runs_returns_list` | Loads logged run, correct keys present |
| `test_load_runs_empty_dir` | Returns empty list when no runs |
| `test_run_id_format` | run_id contains experiment name |

**Status:** ✅ Complete

---

### `tests/test_evaluator.py`
| Test | Description |
|------|-------------|
| `test_perplexity_positive` | Perplexity > 1 for non-zero loss |
| `test_bleu1_range` | BLEU-1 score in [0, 1] |
| `test_evaluate_returns_result` | Full EvalResult with expected fields |
| `test_perfect_bleu` | Identical prediction → BLEU-1 = 1.0 |

**Status:** ✅ Complete

---

### `tests/test_api.py`
| Test | Description |
|------|-------------|
| `test_health` | `GET /api/health` → 200 + `{"status":"ok"}` |
| `test_stats_empty` | `GET /api/stats` → 200 + correct structure |
| `test_runs_empty` | `GET /api/runs` → 200 + `[]` |
| `test_generate` | `POST /api/generate` → 200 + text field |
| `test_train_start` | `POST /api/train` → 200 + `{"status":"started"}` |
| `test_train_conflict` | Second concurrent `POST /api/train` → 409 |
| `test_get_run_not_found` | `GET /api/runs/nonexistent` → 404 |
| `test_delete_run_not_found` | `DELETE /api/runs/nonexistent` → 404 |

**Status:** ✅ Complete

---

## Component Completion Status

| Component | Status | Notes |
|-----------|--------|-------|
| `config.py` | ✅ Done | Pydantic v2, full hyperparameter coverage |
| `data.py` | ✅ Done | HF datasets + demo fallback |
| `trainer.py` | ✅ Done | Simulated loop + GPU code path comments |
| `tracker.py` | ✅ Done | JSON persistence, W&B-compatible interface |
| `evaluator.py` | ✅ Done | Perplexity, BLEU-1/2, EvalResult |
| `utils.py` | ✅ Done | Logging, seeding, formatting |
| `api.py` | ✅ Done | Full REST + SSE |
| `serve.py` | ✅ Done | SPA + API mount |
| `cli.py` | ✅ Done | train / runs / serve commands |
| `ui/index.html` | ✅ Done | Dashboard, Train, Runs, Playground pages |
| `__init__.py` | ✅ Done | Public exports + version |
| `tests/conftest.py` | ✅ Done | Shared fixtures |
| `tests/test_config.py` | ✅ Done | |
| `tests/test_data.py` | ✅ Done | |
| `tests/test_trainer.py` | ✅ Done | |
| `tests/test_tracker.py` | ✅ Done | |
| `tests/test_evaluator.py` | ✅ Done | |
| `tests/test_api.py` | ✅ Done | |
| `pyproject.toml` | ✅ Done | pytest config added |
| `.gitignore` | ✅ Done | |
| `Dockerfile` | ✅ Done | |
| `docker-compose.yml` | ✅ Done | |
| `.env.example` | ✅ Done | |
| `README.md` | ✅ Done | Professional, badges, architecture table |
| `CONTRIBUTING.md` | ✅ Done | Dev setup, conventions, PR guide |

---

## Data Flow

```
CLI (typer)
  │  modelforge train
  │
  ├─► config.TrainingConfig  (validate hyperparams)
  ├─► data.load_dataset      (HF or synthetic)
  ├─► trainer.train          (QLoRA loop / demo)
  └─► tracker.ExperimentTracker.log_run  →  ./logs/<run_id>.json

Web UI (Alpine.js)
  │  POST /api/train
  │
  ├─► api._training_worker (background thread)
  │     ├─► data.load_dataset
  │     ├─► trainer.train (with progress_cb → _train_queue)
  │     └─► tracker.log_run
  │
  └─► GET /api/train/stream  (SSE: progress / complete / error events)
        └─► EventSource in browser  →  live Chart.js update
```

---

## Environment Variables (see `.env.example`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | — | Weights & Biases API key for real experiment tracking |
| `HF_TOKEN` | — | HuggingFace token for gated models (Llama, Gemma) |
| `MODELFORGE_LOG_DIR` | `./logs` | Directory for experiment JSON files |
| `MODELFORGE_OUTPUT_DIR` | `./outputs` | Directory for saved adapters |
| `PORT` | `8080` | Port for the FastAPI server |

---

## Production Upgrade Path

| Demo mode (current) | Production (GPU server) |
|--------------------|------------------------|
| Simulated loss curve | `transformers` + `peft` + `trl` real training |
| Echo generate endpoint | vLLM serving LoRA adapter |
| Local JSON tracking | Weights & Biases full tracking |
| No quantization | 4-bit NF4 BitsAndBytes |
| No model download | `HF_TOKEN` + `AutoModelForCausalLM.from_pretrained` |

Install production extras:
```bash
pip install torch transformers peft trl bitsandbytes wandb vllm unsloth
```
