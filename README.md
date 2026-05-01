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
├── QUICKSTART.md           guided setup and first notebook
├── examples/
│   ├── 01_quickstart.py              marimo quickstart
│   ├── 02_model_parameters_and_hydration.py   map/hydrate knobs
│   ├── 03_cache_back_to_back_and_hydrate.py   disk cache narrative
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

# 4. Open the quickstart notebook
marimo edit examples/01_quickstart.py
```

`maturin develop` must be re-run every time you edit `src/lib.rs`.

## Structured responses (`AiResponse`)

`.ctx.map(model=…)` produces a **`pl.Struct`** column whose schema matches **`pl_ai.AiResponse`**:

- **`status`** — one of `pl_ai.RESPONSE_STATUS_*` (`ok`, `cache_hit`, `budget_exhausted`, `model_error`, `invalid_context`, …).
- **`value`** — decoded model text when complete; otherwise may be empty or null depending on outcome.
- **`cache_key`** — deterministic per row/model row hash (stable across runs).
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
    lf.with_columns(pl_ai.text_context(pl.col("ai").ai.value()).alias("ctx2"))
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

## What the marimo notebook covers

[`examples/01_quickstart.py`](examples/01_quickstart.py):

1. `pl_ai.text_context()` and **`is_context_dtype`**
2. `.ctx.preview()` / `.ctx.estimate_tokens()`
3. `.ctx.map` → **`AiResponse`**, unpacked with `.ai.value()` / `.ai.status()`
4. Budget demo (`max_requests`) and hydrate to finish partial runs
5. Chained maps (each step wraps **`.ai.value()`** before the next `text_context`)
6. Image context struct inspection
7. Optional OpenAI-backed cells (`OPENAI_API_KEY`)
8. `pl_ai.context()` convenience helper

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
