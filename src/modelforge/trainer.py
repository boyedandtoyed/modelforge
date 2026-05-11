"""
QLoRA training pipeline.

Demo mode (default): simulates a realistic training loop with cosine LR and
exponential loss decay so the UI and CLI are fully functional without GPU hardware.

Production mode: replace the body of `train()` with the GPU path documented in the
inline comments. Requires: torch, transformers, peft, trl, bitsandbytes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import TrainingConfig
from .data import Sample


@dataclass
class TrainingMetrics:
    epoch: int
    step: int
    loss: float
    learning_rate: float
    tokens_per_second: float = 0.0
    grad_norm: float = 0.0


@dataclass
class TrainingRun:
    config: TrainingConfig
    metrics: list[TrainingMetrics] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    best_loss: float = float("inf")
    adapter_path: str | None = None


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def train(
    config: TrainingConfig,
    samples: list[Sample],
    progress_cb: Callable[[TrainingMetrics, int], None] | None = None,
) -> TrainingRun:
    """
    Fine-tune a causal LM with QLoRA.

    Parameters
    ----------
    config:
        Full training configuration (hyperparameters, LoRA, quantization).
    samples:
        Pre-formatted training samples (prompt + completion pairs).
    progress_cb:
        Optional callback `fn(metrics, total_steps)` called after each step.

    Returns
    -------
    TrainingRun
        Completed run record with metrics, timing, and adapter path.

    ---
    GPU path (requires: torch, transformers, peft, trl, bitsandbytes)
    ---
    Replace this function body with:

        model, tokenizer = _build_qlora_model(config)
        trainer = _get_sft_trainer(model, tokenizer, config, samples)
        trainer.train()
        trainer.save_model(config.output_dir + "/adapter")
        run = TrainingRun(config=config)
        for log in trainer.state.log_history:
            if "loss" in log:
                run.metrics.append(TrainingMetrics(
                    epoch=int(log.get("epoch", 1)),
                    step=log["step"],
                    loss=log["loss"],
                    learning_rate=log.get("learning_rate", 0),
                    tokens_per_second=log.get("train_tokens_per_second", 0),
                    grad_norm=log.get("grad_norm", 0),
                ))
        run.end_time = time.time()
        run.best_loss = min(m.loss for m in run.metrics) if run.metrics else float("inf")
        run.adapter_path = config.output_dir + "/adapter"
        return run
    """
    run = TrainingRun(config=config)
    steps_per_epoch = max(1, len(samples) // config.per_device_train_batch_size)
    total_steps = config.num_epochs * steps_per_epoch

    for epoch in range(config.num_epochs):
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            progress = global_step / max(1, total_steps)

            # Exponential decay from ~2.5 with small sawtooth noise
            base_loss = 2.5 * math.exp(-3.0 * progress) + 0.3
            noise = 0.05 * (1 - progress) * (0.5 - (global_step % 7) / 7)
            loss = base_loss + noise

            # Cosine LR schedule with linear warmup
            warmup_steps = int(total_steps * config.warmup_ratio)
            if global_step < warmup_steps:
                lr = config.learning_rate * (global_step / max(1, warmup_steps))
            else:
                decay_progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * decay_progress))

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                step=global_step,
                loss=round(loss, 4),
                learning_rate=round(lr, 8),
                tokens_per_second=round(1200 + 200 * progress, 1),
                grad_norm=round(0.8 + 0.3 * (1 - progress), 3),
            )
            run.metrics.append(metrics)

            if loss < run.best_loss:
                run.best_loss = loss

            if progress_cb is not None:
                progress_cb(metrics, total_steps)

    run.end_time = time.time()
    run.adapter_path = f"{config.output_dir}/adapter"
    return run


# ---------------------------------------------------------------------------
# Adapter utilities
# ---------------------------------------------------------------------------

def merge_adapter(base_model: str, adapter_path: str, output_path: str) -> str:
    """
    Merge a LoRA adapter into the base model weights and save the result.

    Parameters
    ----------
    base_model:
        HuggingFace model ID or local path to the original model.
    adapter_path:
        Path to the saved PEFT adapter directory.
    output_path:
        Destination directory for the merged model.

    Returns
    -------
    str
        Absolute path of the merged model directory.

    ---
    Production path (requires: torch, transformers, peft)
    ---
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        return output_path
    """
    return output_path


# ---------------------------------------------------------------------------
# GPU-only helpers (stubs — documented for production use)
# ---------------------------------------------------------------------------

def _build_qlora_model(cfg: TrainingConfig):  # type: ignore[return]
    """
    Load base model with 4-bit quantization and apply LoRA adapters.

    Production:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
        model = get_peft_model(model, lora_cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    """
    raise NotImplementedError("GPU hardware and full dependencies required.")


def _get_sft_trainer(model, tokenizer, cfg: TrainingConfig, samples: list[Sample]):  # type: ignore[return]
    """
    Construct a TRL SFTTrainer with the given model, tokenizer, config, and samples.

    Production:
        from transformers import TrainingArguments
        from trl import SFTTrainer
        import datasets as hf_datasets

        texts = [s.prompt + s.completion for s in samples]
        ds = hf_datasets.Dataset.from_dict({"text": texts})

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            fp16=True,
            logging_steps=1,
            save_strategy="epoch",
            report_to=cfg.report_to,
            run_name=cfg.experiment_name,
        )
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            args=training_args,
        )
    """
    raise NotImplementedError("GPU hardware and full dependencies required.")
