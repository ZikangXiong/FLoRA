# FLoRA

Clean network implementation for learning temporal-logic-style scoring rules from trajectory predicates.

This repository does not include the full nuPlan pipeline. It contains the core learnable network, extraction utilities, training helpers, tests, and the project website/docs.

## Contents

- `flora/`: network, differentiable logic operators, extraction, and training utilities
- `tests/`: unit tests for the core implementation
- `examples/`: minimal usage example
- `docs/`: website and documentation source

## Install

```bash
uv sync --all-extras
```

## Test

```bash
uv run pytest tests/
```
