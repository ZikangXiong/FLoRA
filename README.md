# FLoRA: Learnable Temporal Logic Network

Implementation of the learnable logic structure from [FLoRA: A Framework for Learning Scoring Rules in Autonomous Driving Planning Systems](https://arxiv.org/abs/2502.11352) (IEEE RA-L).

Unlike traditional temporal logic tools where the user specifies the formula, FLoRA **learns both the formula structure and parameters from data**. After training, the soft network is discretized into an interpretable temporal logic formula.

## How It Works

```
Input: (B, N, T)  predicate signals — batch x predicates x timesteps
         │
         ▼
  ┌─────────────────────┐
  │   PredicateLayer    │  (optional) learnable threshold & scale per predicate
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐
  │  TemporalLayer x K  │  per predicate: softmax gate selects {identity, G, F}
  │                     │  + tanh negation gate (smooth ±1)
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐
  │ PropositionalLayer  │  pairs consecutive predicates with learned {AND, OR, IMPLIES}
  └─────────┬───────────┘
            ▼
  ┌─────────────────────┐
  │  AggregationLayer   │  combines all pairs with learned {AND, OR}
  └─────────┬───────────┘
            ▼
     Score at t=0 → (B,)
```

During training all gates are **soft** (softmax-blended) for differentiability. After training, gates are **discretized** (argmax) and the formula is extracted via sympy.

Temporal operators use **finite-trace quantitative semantics**: `G(P)(t) = min over suffix [t:T]`, `F(P)(t) = max over suffix [t:T]`. The time dimension is preserved through all layers, so nested compositions like `G(F(P))` work naturally — the boundary effect at the last timestep is standard for finite-trace STL.

## Setup

Requires Python 3.10+ and a CUDA-capable GPU.

### Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync --all-extras
```

### Run tests

```bash
uv run pytest tests/
```

## Quick Start

```python
import torch
from flora import DEVICE
from flora.network import LogicNetwork
from flora.train import generate_data_G_and_F, train

# Generate data for G(P0) ∧ F(P1)
pos, neg = generate_data_G_and_F(1000, 2, 10, g_idx=0, f_idx=1)

# Train
net = LogicNetwork(n_predicates=2, n_temporal_layers=1, hardness=10.0).to(DEVICE)
train(net, pos, neg, lr=0.05, epochs=300, batch_size=64)

# Extract formula
print(net.extract(["safe_distance", "goal_reached"]))
```

See `examples/learn_formula.ipynb` for the full walkthrough with visualizations.

## Project Structure

```
flora/
├── __init__.py     # DEVICE auto-detection (CUDA/CPU)
├── ops.py          # Differentiable primitives: soft_globally, soft_finally, soft_and, soft_or, ...
├── network.py      # PredicateLayer, TemporalLayer, PropositionalLayer, AggregationLayer, LogicNetwork
├── extract.py      # Discretize gates → temporal simplification (De Morgan, idempotence) → sympy CNF
└── train.py        # Data generators + contrastive training with hardness annealing
tests/              # 75 tests
examples/
└── learn_formula.ipynb
```
