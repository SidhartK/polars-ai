# polars_ai

Structured AI inference as Polars expressions: each row yields an **AiResponse** struct (status, text value, errors, timestamps, deterministic cache keys) plus optional budgets, hydration, and a disk cache.

For a guided setup, see [`QUICKSTART.md`](QUICKSTART.md).

## Prerequisites

| Tool | Install |
|------|---------|
| Rust toolchain | `curl https://sh.rustup.rs \| sh` |
| Python 3.9+ | system / pyenv / conda |
| maturin | `pip install maturin` |

## Directory layout

```
polars_ai/
├── Cargo.toml              Rust dependencies
├── pyproject.toml          Python build config (maturin)
├── QUICKSTART.md           guided setup and notebook sequence
├── examples/
│   ├── 01_contexts_and_batches.py          context atoms and batches
│   ├── 02_map_operations.py                vanilla map operations
│   ├── 03_hydration_real_dataset.py        hydration with real text data
│   ├── 04_caching_and_reproducibility.py   disk cache replay
│   └── static/
│       └── example.jpg     sample image used by the notebook
├── src/
│   └── lib.rs              Rust plugins (Tokio async, bounded concurrency)
└── polars_ai/              Python package
    ├── __init__.py         public API + namespace registration
    ├── types.py            AiModelContext, AiResponse, status constants
    ├── constructors.py     text_context(), image_context(), context()
    ├── model.py            AiModel ABC + FakeModel
    ├── namespace.py        @register_expr_namespace("ctx")
    └── response_namespace.py   @register_expr_namespace("ai")
```

## Setup and run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install maturin
pip install maturin

# 3. Compile the Rust plugin and install it into the venv
maturin develop --extras examples

# 4. Open the first notebook
marimo edit examples/01_contexts_and_batches.py
```

`maturin develop` must be re-run every time you edit `src/lib.rs`.

## Structured responses (`AiResponse`)

## Context Atoms And Batches

`AiModelContext` is the row-level context atom: a semi-opaque `Struct[_type, _value, _mime, _meta]` that can render as text or image provider content. `ContextBatch` is a first-class grouped context represented as `List[AiModelContext]`.

```python
batched = (
    pl.LazyFrame({"topic": ["a", "a"], "msg": ["one", "two"]})
    .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
    .group_by("topic", maintain_order=True)
    .agg(pl.col("ctx").ctx.batch().alias("ctx_batch"))
)

reduced = batched.with_columns(
    pl.col("ctx_batch")
    .ctxbatch.reduce_text(text_separator="\n\n", number_text_items=True)
    .alias("ctx_reduced")
)
```

The intended flow is:

1. `ContextAtom -> ContextBatch` with `.ctx.batch()` in grouped aggregations.
2. `ContextBatch -> ContextAtom(text)` with `.ctxbatch.reduce_text(...)` when a provider needs one text prompt.
3. `ContextAtom | ContextBatch -> AiResponse` with `.ctx.map(...)` or `.ctxbatch.map(...)`.
4. `AiResponse -> ContextAtom` with `.ai.to_context()` for chained model calls.

`.ctx.map(model=…)` produces a **`pl.Struct`** column whose schema matches **`pl_ai.AiResponse`**:

- **`status`** — one of `pl_ai.RESPONSE_STATUS_*` (`ok`, `cache_hit`, `budget_exhausted`, `model_error`, `invalid_context`, …).
- **`value`** — decoded model text when complete; otherwise may be empty or null depending on outcome.
- **`cache_key`** — deterministic per row/model row hash (stable across runs).
- **`input_tokens` / `output_tokens` / `total_tokens` / `cost_usd`** — telemetry returned with every response and persisted in cache entries when caching is enabled.
- **`error`** / **`attempts`** / timestamps — bookkeeping for debugging and dashboards.

Unpack with the **`.ai`** namespace on response columns:

```python
import polars as pl
import polars_ai as pl_ai

lf = (
    pl.LazyFrame({"review": ["Great!", "Bad."]})
    .with_columns(pl_ai.text_context(pl.col("review")).alias("ctx"))
    .with_columns(pl.col("ctx").ctx.map(model=pl_ai.FakeModel(tag="demo")).alias("ai"))
)

df = lf.collect()
decoded = df.select(
    pl.col("ai").ai.status().alias("status"),
    pl.col("ai").ai.value().alias("answer"),
)

# downstream map: wrap the decoded string as a NEW context column
step2_model = pl_ai.FakeModel(prompt="Second pass: {value}", tag="step2")
piped = (
    lf.with_columns(pl.col("ai").ai.to_context().alias("ctx2"))
    .with_columns(pl.col("ctx2").ctx.map(model=step2_model).alias("step2"))
)
piped.collect()
```

## Budgets

`.ctx.map` accepts guardrails interpreted by the Rust engine:

| Parameter | Meaning |
|-----------|---------|
| `max_requests` | Cap successful model completions (excluding cache hits); remaining rows → `budget_exhausted`. |
| `max_tokens` | Estimated input-token budget (~ `ceil(chars/4)` for text); exhaustion applies in row order once the cap is exceeded. |
| `max_concurrency` | Upper bound on in-flight parallel model calls (`>= 1` or `None` for default). |
| `rate_limit_per_second` | Optional pacing (Governor-backed in the Rust layer). |

**Important:** Lazy Polars does not bill like a prepaid API meter. Projection or predicate pushdown can change which rows compute; budgets apply to the rows actually executed. Treat budgets as coarse safety rails rather than cryptographic spend caps.

## Hydration

If a pipeline stopped early (`budget_exhausted`, etc.), re-run **`pl.col("out").ai.hydrate(ctx=pl.col("ctx"), model=model, …)`**. Already-terminal rows (**`ok`**, **`cache_hit`**, **`invalid_context`**) are preserved; only incomplete rows consume the fresh budget against the disk cache → model precedence.

Hydration accepts the same budget and cache knobs as `.ctx.map`.

## Disk cache

Enable with `cache=True` and optional **`cache_path`** (defaults to `./__polars_ai_cache__`; see **`pl_ai.DEFAULT_CACHE_FOLDER`**). Entries are keyed by **`cache_key`**. Cached rows reuse values with **`RESPONSE_STATUS_CACHE_HIT`** without spending `max_requests` on the provider.

Clear the cache folder to force fresh calls while keeping deterministic keys for debugging.

## Cache keys (`ctx.cache_key`)

`pl.col("ctx").ctx.cache_key(model=my_model)` materializes each row's stable fingerprint so you can join, duplicate-check, or prefill cache metadata without calling the provider.

## What the marimo notebooks cover

The examples are ordered as a four-notebook learning path:

1. [`examples/01_contexts_and_batches.py`](examples/01_contexts_and_batches.py) introduces `AiModelContext`, `ContextBatch`, `.ctx.preview()`, `.ctx.estimate_tokens()`, `.ctx.batch()`, and `.ctxbatch.reduce_text(...)`.
2. [`examples/02_map_operations.py`](examples/02_map_operations.py) introduces `.ctx.map(...)`, `.ctxbatch.map(...)`, `AiResponse` accessors, budgets, chained maps, and cache-key preflight checks.
3. [`examples/03_hydration_real_dataset.py`](examples/03_hydration_real_dataset.py) uses the UCI SMS Spam Collection to show partial runs and `.ai.hydrate(...)` on `budget_exhausted` rows.
4. [`examples/04_caching_and_reproducibility.py`](examples/04_caching_and_reproducibility.py) shows `cache=True`, replaying rows as `cache_hit` after rebuilding the dataframe, cache misses for changed inputs, and hydrate-with-cache.

## Extending with a real model

Subclass `AiModel` and implement `model_config` as JSON consumed by Rust:

```python
from polars_ai import AiModel
import json


class OpenAIModel(AiModel):
    def __init__(self, prompt: str, model: str = "gpt-4o-mini"):
        self.prompt = prompt
        self.model = model

    @property
    def model_config(self) -> str:
        return json.dumps({"prompt": self.prompt, "model": self.model})
```

Wire HTTP in `src/lib.rs` (`reqwest`) where `fake_model_call` substitutes today, using the JSON config to dispatch per provider.
