# Quickstart

Install `polars-ai` locally and open [`examples/01_contexts_and_batches.py`](examples/01_contexts_and_batches.py), the first marimo notebook in a four-part sequence covering context atoms, context batches, map operations, hydration, and caching.

## Prerequisites

- Python 3.9+
- Rust toolchain
- `maturin`

Install `maturin` into your environment if you do not already have it:

```bash
pip install maturin
```

## Set up a local environment

From `polars-ai/` (repository root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --extras examples
```

`maturin develop` compiles the Rust extension and installs `polars-ai` into the active virtual environment. Re-run it after editing `src/lib.rs`.

You can optionally install dev tooling (pytest, ruff, …) via `pip install -e ".[dev]"`.

## Open the notebook

```bash
marimo edit examples/01_contexts_and_batches.py
```

## Notebook sequence

1. [`examples/01_contexts_and_batches.py`](examples/01_contexts_and_batches.py): create and inspect **`AiModelContext`** values, group them into **`ContextBatch`** values, and reduce batches back to text contexts.
2. [`examples/02_map_operations.py`](examples/02_map_operations.py): map context columns into **`AiResponse`** structs, unpack `.ai.*` fields, use budgets, chain maps, and map grouped batches.
3. [`examples/03_hydration_real_dataset.py`](examples/03_hydration_real_dataset.py): use a small UCI SMS Spam Collection sample to stop a run with `budget_exhausted` rows and finish it with **`.ai.hydrate(...)`**.
4. [`examples/04_caching_and_reproducibility.py`](examples/04_caching_and_reproducibility.py): warm a disk cache, rebuild the dataframe, replay cached rows with `max_requests=0`, and hydrate cache misses.

## Behavioral notes

- **Budget vs optimizer:** Limits apply to whichever rows Lazy Polars executes after optimization. Projection may skip columns or rows unexpectedly; budgets are safeguards, not a guaranteed bill.
- **Cache folder:** Relative **`./__polars_ai_cache__`** unless you override **`cache_path`**. Cached hits report **`RESPONSE_STATUS_CACHE_HIT`**.
- **Context state:** atoms are stored as `Struct`; batches are stored as `List[Struct]`; rendered provider content is internal to model invocation.

## Provider calls

The notebooks rely on **`FakeModel`** by default, so no provider API key is required.
