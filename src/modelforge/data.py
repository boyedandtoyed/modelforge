"""Dataset loading and preprocessing for instruction fine-tuning."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


ALPACA_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

CHAT_TEMPLATE = """\
<|system|>
You are a helpful assistant.
<|user|>
{instruction}
<|assistant|>
{output}"""


@dataclass
class Sample:
    prompt: str
    completion: str
    tokens: int = 0


def format_alpaca(row: dict[str, Any]) -> Sample:
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", "")
    prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output="")
    return Sample(prompt=prompt, completion=output, tokens=len(prompt.split()) + len(output.split()))


def format_chat(row: dict[str, Any]) -> Sample:
    instruction = row.get("instruction", row.get("prompt", ""))
    output = row.get("output", row.get("response", ""))
    prompt = CHAT_TEMPLATE.format(instruction=instruction, output="")
    return Sample(prompt=prompt, completion=output, tokens=len(prompt.split()) + len(output.split()))


def load_dataset(dataset_id: str, split: str = "train", max_samples: int | None = None) -> list[Sample]:
    """
    Load and format a dataset.
    In production with transformers/datasets installed:
        from datasets import load_dataset as hf_load
        ds = hf_load(dataset_id, split=split)
    """
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load(dataset_id, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        formatter = format_chat if "chat" in dataset_id.lower() else format_alpaca
        return [formatter(row) for row in ds]
    except Exception:
        # Demo fallback: return synthetic samples
        return _synthetic_samples(max_samples or 100)


def _synthetic_samples(n: int) -> list[Sample]:
    instructions = [
        "Explain what a transformer model is.",
        "Write a Python function to sort a list.",
        "What is the difference between precision and recall?",
        "Summarize the concept of attention in neural networks.",
        "How does backpropagation work?",
    ]
    outputs = [
        "A transformer model is a neural network architecture based on self-attention mechanisms...",
        "def sort_list(lst): return sorted(lst)",
        "Precision measures true positives over predicted positives. Recall measures true positives over actual positives.",
        "Attention allows a model to weigh the importance of different tokens when computing representations.",
        "Backpropagation computes gradients of the loss with respect to model parameters using the chain rule.",
    ]
    samples = []
    for i in range(n):
        idx = i % len(instructions)
        samples.append(format_alpaca({"instruction": instructions[idx], "input": "", "output": outputs[idx]}))
    return samples
