"""Tests for the Evaluator module."""
from __future__ import annotations

import math

from modelforge.data import Sample, _synthetic_samples
from modelforge.evaluator import EvalResult, Evaluator, _ngrams


def test_perplexity_positive(tiny_samples):
    ev = Evaluator()
    ppl = ev.compute_perplexity(tiny_samples)
    assert ppl > 1.0


def test_perplexity_empty():
    ev = Evaluator()
    assert ev.compute_perplexity([]) == float("inf")


def test_bleu1_range(tiny_samples):
    ev = Evaluator()
    score = ev.compute_bleu(tiny_samples, n=1)
    assert 0.0 <= score <= 1.0


def test_bleu2_range(tiny_samples):
    ev = Evaluator()
    score = ev.compute_bleu(tiny_samples, n=2)
    assert 0.0 <= score <= 1.0


def test_perfect_bleu():
    text = "the quick brown fox"
    sample = Sample(prompt=text, completion=text, tokens=4)
    ev = Evaluator()
    score = ev.compute_bleu([sample], n=1)
    assert score == 1.0


def test_evaluate_returns_result(tiny_samples):
    ev = Evaluator()
    result = ev.evaluate(tiny_samples)
    assert isinstance(result, EvalResult)
    assert result.num_samples == len(tiny_samples)
    assert result.perplexity > 1.0
    assert 0.0 <= result.bleu1 <= 1.0
    assert 0.0 <= result.bleu2 <= 1.0
    assert result.eval_time >= 0.0


def test_eval_result_avg_loss_consistent(tiny_samples):
    ev = Evaluator()
    result = ev.evaluate(tiny_samples)
    assert abs(result.avg_loss - math.log(result.perplexity)) < 1e-3


def test_ngrams_unigrams():
    tokens = ["a", "b", "c", "a"]
    counts = _ngrams(tokens, 1)
    assert counts[("a",)] == 2
    assert counts[("b",)] == 1


def test_ngrams_bigrams():
    tokens = ["a", "b", "c"]
    counts = _ngrams(tokens, 2)
    assert counts[("a", "b")] == 1
    assert counts[("b", "c")] == 1
    assert len(counts) == 2
