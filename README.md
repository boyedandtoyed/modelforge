# ModelForge

Production-grade LLM fine-tuning toolkit with QLoRA, experiment tracking, and model serving.

Supports instruction fine-tuning on any HuggingFace model/dataset. Full QLoRA pipeline with 4-bit quantization, LoRA adapters, cosine LR scheduling, and W&B experiment tracking.

## Quick Start

```bash
pip install -e .

# Fine-tune (demo mode — GPU + full deps for real training)
modelforge train --model meta-llama/Llama-2-7b-hf --dataset tatsu-lab/alpaca --epochs 3

# List runs
modelforge runs

# Serve model
modelforge serve --adapter ./outputs/adapter --port 8080
```

## Full Dependencies (GPU)

```bash
pip install torch transformers peft trl bitsandbytes wandb vllm
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Fine-tuning | QLoRA (PEFT + bitsandbytes) |
| Accelerated training | Unsloth |
| Dataset | HuggingFace datasets |
| Tracking | Weights & Biases |
| Serving | vLLM |
| CLI | Typer + Rich |
| API | FastAPI + Pydantic v2 |
