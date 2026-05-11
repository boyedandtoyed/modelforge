# Contributing to ModelForge

Thank you for your interest in contributing. This document covers the development setup, coding conventions, and pull request process.

---

## Development Setup

```bash
git clone <repo-url> && cd modelforge

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify the setup
python3 -m pytest          # should pass 31 tests
modelforge --help          # should print CLI help
```

### Environment variables

```bash
cp .env.example .env
# Fill in HF_TOKEN and/or WANDB_API_KEY as needed
```

---

## Project Structure

See [Code_map.md](Code_map.md) for a full map of every module, class, and function.

---

## Running Tests

```bash
# All tests
python3 -m pytest

# Single module
python3 -m pytest tests/test_api.py -v

# With coverage
python3 -m pytest --cov=modelforge --cov-report=term-missing
```

Tests must pass before any PR can be merged. No test should rely on network access, GPU hardware, or external services — use the demo/synthetic fallback paths.

---

## Code Conventions

- **Python 3.11+** with `from __future__ import annotations` in every module.
- **Pydantic v2** for all data models.
- **Type hints** on every public function and method.
- **No inline comments** explaining *what* code does — only *why* (non-obvious invariants, workarounds, production vs. demo branches).
- **No docstrings** beyond a one-line module docstring.
- **Trailing whitespace** stripped; files end with a newline.
- **Imports**: stdlib → third-party → local, one blank line between groups.

---

## Adding a Feature

1. Open an issue describing the feature and motivation before writing code.
2. Branch from `master`: `git checkout -b feat/my-feature`.
3. Add or extend tests in `tests/` — new public functions need at least one test.
4. Update `Code_map.md` if you add a new module, class, or public function.
5. Run the full test suite locally.
6. Open a pull request with a clear description of what changed and why.

---

## GPU / Production Path Changes

ModelForge runs in demo mode by default. If you extend the GPU training path in `trainer.py` or the vLLM serve path, document the production code as an inline comment (same pattern as the existing stubs) so the demo fallback remains functional without GPU hardware.

---

## Pull Request Checklist

- [ ] All 31+ tests pass (`python3 -m pytest`)
- [ ] New code has corresponding tests
- [ ] `Code_map.md` updated if public API changed
- [ ] No secrets or model weights committed
- [ ] PR description explains *why*, not just *what*
