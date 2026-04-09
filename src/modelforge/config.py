"""Training configuration with full QLoRA/Unsloth support."""
from __future__ import annotations
from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    r: int = Field(16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha scaling")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Transformer modules to apply LoRA to",
    )
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    bias: str = Field("none", description="none | all | lora_only")
    task_type: str = "CAUSAL_LM"


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class TrainingConfig(BaseModel):
    # Model
    base_model: str = Field("meta-llama/Llama-2-7b-hf", description="HuggingFace model ID")
    output_dir: str = "./outputs"

    # Data
    dataset: str = Field("tatsu-lab/alpaca", description="HuggingFace dataset ID")
    dataset_split: str = "train"
    max_samples: int | None = None
    max_seq_length: int = 2048

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # QLoRA
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    # Experiment tracking
    experiment_name: str = "modelforge-run"
    report_to: str = "wandb"  # wandb | tensorboard | none

    # Serving
    serve_port: int = 8080
    max_new_tokens: int = 512
    temperature: float = 0.7
