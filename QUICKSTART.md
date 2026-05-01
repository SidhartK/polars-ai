# Quickstart

Install `polars-ai` locally and open [`examples/01_quickstart.py`](examples/01_quickstart.py), a marimo notebook that walks through contexts, **`AiResponse`** structs (`.ai.*` accessors), budgets, hydrate, caching, and optional OpenAI.

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
marimo edit examples/01_quickstart.py
```

## What you will practice

1. **`pl_ai.text_context()`** (`AiModelContext` struct columns) and **`pl_ai.is_context_dtype()`**
2. **`.ctx.preview()`** / **`.ctx.estimate_tokens()`** (pure Polars helpers)
3. **`.ctx.map(model=..., max_requests=..., max_tokens=..., cache=...)`** returning **`AiResponse`**
4. **`.ai.value()`**, **`.ai.status()`**, **`.ai.cache_key()`** on mapped columns
5. **Hydration**: **`.ai.hydrate(ctx=..., model=..., ...budgets)`** to finish **`budget_exhausted`** rows
6. Chaining pipelines: rebuild context from **`pl_ai.text_context(pl_col.ai.value())`** between stages
7. Image contexts (`pl_ai.image_context`) and **`pl_ai.context()`**
8. Optional OpenAI cells when **`OPENAI_API_KEY`** is set

## Behavioral notes

- **Budget vs optimizer:** Limits apply to whichever rows Lazy Polars executes after optimization. Projection may skip columns or rows unexpectedly; budgets are safeguards, not a guaranteed bill.
- **Cache folder:** Relative **`./__polars_ai_cache__`** unless you override **`cache_path`**. Cached hits report **`RESPONSE_STATUS_CACHE_HIT`**.

## Optional OpenAI cells

```bash
export OPENAI_API_KEY=...
marimo edit examples/01_quickstart.py
```

Without an API key, the notebook skips live calls and relies on **`FakeModel`**.
