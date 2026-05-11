"""Post-training evaluation: perplexity, n-gram metrics, and result aggregation."""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

from .data import Sample


@dataclass
class EvalResult:
    perplexity: float
    avg_loss: float
    bleu1: float
    bleu2: float
    num_samples: int
    eval_time: float = field(default=0.0)

    def __str__(self) -> str:
        return (
            f"EvalResult(perplexity={self.perplexity:.2f}, "
            f"avg_loss={self.avg_loss:.4f}, "
            f"bleu1={self.bleu1:.4f}, bleu2={self.bleu2:.4f}, "
            f"n={self.num_samples})"
        )


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Return a Counter of all n-grams in *tokens*."""
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


class Evaluator:
    """
    Lightweight post-training evaluator.

    In production with a loaded model + tokenizer replace the demo methods with:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        perplexity = math.exp(loss)
    """

    def __init__(self, model_path: str = "demo") -> None:
        self.model_path = model_path
        self._is_demo = model_path == "demo"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_perplexity(self, samples: list[Sample]) -> float:
        """
        Mean per-token perplexity across *samples*.

        Demo: derives an analytic perplexity from token counts (length-
        weighted entropy proxy) so the number is always > 1 and shrinks
        as samples grow longer.

        Production path (requires GPU + transformers):
            import torch
            total_loss, total_tokens = 0.0, 0
            for s in samples:
                enc = tokenizer(s.prompt + s.completion, return_tensors="pt").to(device)
                with torch.no_grad():
                    loss = model(**enc, labels=enc["input_ids"]).loss.item()
                total_loss += loss * enc["input_ids"].shape[1]
                total_tokens += enc["input_ids"].shape[1]
            return math.exp(total_loss / total_tokens)
        """
        if not samples:
            return float("inf")

        if not self._is_demo:
            raise RuntimeError("Non-demo model path requires transformers + torch.")

        # Analytic proxy: perplexity ≈ e^(1 / avg_token_count * scaling_factor)
        avg_tokens = sum(s.tokens for s in samples) / len(samples)
        scaling = max(1.5, 8.0 - 0.05 * avg_tokens)
        loss_proxy = scaling / max(1, avg_tokens ** 0.5)
        return round(math.exp(loss_proxy), 4)

    def compute_bleu(self, samples: list[Sample], n: int = 1) -> float:
        """
        Corpus-level BLEU-n score.  Uses the reference = completion and
        hypothesis = prompt tail (last n words) as a demo stand-in.
        Returns a value in [0, 1].

        Production path: pass model-generated completions as hypotheses.
        """
        if not samples or n < 1:
            return 0.0

        total_match = 0
        total_hyp = 0

        for sample in samples:
            ref_tokens = sample.completion.lower().split()
            hyp_tokens = sample.prompt.lower().split()

            ref_ngrams = _ngrams(ref_tokens, n)
            hyp_ngrams = _ngrams(hyp_tokens, n)

            clipped = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
            total_match += clipped
            total_hyp += max(1, len(hyp_tokens) - n + 1)

        if total_hyp == 0:
            return 0.0
        return round(total_match / total_hyp, 6)

    def evaluate(self, samples: list[Sample]) -> EvalResult:
        """Run a full evaluation pass and return an EvalResult."""
        t0 = time.time()

        perplexity = self.compute_perplexity(samples)
        avg_loss = round(math.log(max(1.0001, perplexity)), 4)
        bleu1 = self.compute_bleu(samples, n=1)
        bleu2 = self.compute_bleu(samples, n=2)

        return EvalResult(
            perplexity=perplexity,
            avg_loss=avg_loss,
            bleu1=bleu1,
            bleu2=bleu2,
            num_samples=len(samples),
            eval_time=round(time.time() - t0, 3),
        )
