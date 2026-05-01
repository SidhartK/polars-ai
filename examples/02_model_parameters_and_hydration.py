"""Marimo notebook: every ``.ctx.map`` / ``.ai.hydrate`` knob + hydration patterns."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai
    from polars_ai import FakeModel

    return FakeModel, Path, mo, pl, pl_ai, shutil


@app.cell
def _(mo):
    mo.md("""
    # 02 — Model call parameters and hydration

    This notebook walks through **every keyword** accepted by **`pl.col("ctx").ctx.map(...)`**
    and **`pl.col("ai").ai.hydrate(ctx=..., ...)`**:

    | Parameter | Role |
    |-----------|------|
    | **`model`** | `AiModel` instance (`FakeModel` here). |
    | **`max_requests`** | Cap on *new* provider completions (cache hits skip this budget). |
    | **`max_tokens`** | Cumulative rough input-token ceiling (same heuristic as `.ctx.estimate_tokens()`). |
    | **`max_concurrency`** | Tokio parallelism cap inside the Rust plugin (`>= 1` or `None`). |
    | **`rate_limit_per_second`** | Local pacing; **`0`** disables the limiter. |
    | **`cache`** / **`cache_path`** | Optional JSONL disk cache under `cache_entries.jsonl`. |

    **Hydration** reuses the same knobs: pass the prior **`AiResponse`** column plus the original **`ctx`** so
    incomplete rows can be retried while **terminal** rows (`ok`, `cache_hit`, …) are copied through unchanged.

    A scratch cache folder **`__02_param_demo_cache__/`** is created next to this file (gitignored) and removed at startup.
    """)
    return


@app.cell
def _(Path, shutil):
    DEMO_CACHE = Path(__file__).resolve().parent / "__02_param_demo_cache__"
    shutil.rmtree(DEMO_CACHE, ignore_errors=True)
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    return (DEMO_CACHE,)


@app.cell
def _(FakeModel, mo, pl, pl_ai):
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "label": ["A", "B", "C", "D", "E", "F"],
            "note": [
                "aa",  # short — low token estimate
                "bbbb",
                "cccccccc",
                "dddddddddddd",
                "eeeeeeeeeeeeeeee",
                "ffffffffffffffffffffffff",  # long — pushes token budget
            ],
        }
    ).with_columns(
        pl_ai.text_context(pl.col("note"), format_str="Item {value}").alias("ctx"),
    )

    m_requests = FakeModel(prompt="Req {value}", tag="demo-requests")
    m_tokens = FakeModel(prompt="Tok {value}", tag="demo-tokens")
    m_tuned = FakeModel(prompt="Tune {value}", tag="demo-tune")

    mo.vstack(
        [
            mo.md("## 1. Sample frame + three tagged `FakeModel`s"),
            df.select("id", "label", "note", pl.col("ctx").ctx.estimate_tokens().alias("est_tok")),
        ]
    )
    return df, m_requests, m_tokens, m_tuned


@app.cell
def _(df, m_requests, m_tokens, mo, pl):
    keys = df.lazy().select(
        "id",
        pl.col("ctx").ctx.cache_key(m_requests).alias("k_requests"),
        pl.col("ctx").ctx.cache_key(m_tokens).alias("k_tokens"),
    ).collect()

    mo.vstack(
        [
            mo.md("""
    ## 2. `ctx.cache_key(model)` — stable per row + model JSON

    Same context row, **different** `model_config` ⇒ **different** keys (used by the disk cache and debugging).
            """),
            keys,
        ]
    )
    return


@app.cell
def _(df, m_requests, mo, pl):
    capped = (
        df.lazy()
        .with_columns(
            pl.col("ctx").ctx.map(model=m_requests, max_requests=2).alias("ai"),
        )
        .collect()
    )

    after_hydrate = (
        capped.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(ctx=pl.col("ctx"), model=m_requests, max_requests=10)
            .alias("ai"),
        )
        .collect()
    )

    def _show(d: pl.DataFrame) -> pl.DataFrame:
        return d.select(
            "id",
            "label",
            pl.col("ai").ai.status().alias("status"),
            pl.col("ai").ai.value().str.slice(0, 48).alias("value_head"),
        )

    mo.vstack(
        [
            mo.md("""
    ## 3. `max_requests` + hydrate

    First pass: **`max_requests=2`** ⇒ first two rows **`ok`**, the rest **`budget_exhausted`**.

    Second pass: **`.ai.hydrate(..., max_requests=10)`** fills the remaining rows without touching the
    already-terminal first two.
            """),
            mo.md("### After `ctx.map`"),
            _show(capped),
            mo.md("### After `ai.hydrate`"),
            _show(after_hydrate),
        ]
    )
    return


@app.cell
def _(df, m_tokens, mo, pl):
    tok_run = (
        df.head(4)
        .lazy()
        .with_columns(
            pl.col("ctx").ctx.map(model=m_tokens, max_tokens=4).alias("ai"),
        )
        .collect()
    )

    tok_hydrated = (
        tok_run.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(ctx=pl.col("ctx"), model=m_tokens, max_tokens=20)
            .alias("ai"),
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
    ## 4. `max_tokens` + hydrate

    Rough rule: **`ceil(len(text) / 4)`** per row (see **`est_tok`** above). With **`max_tokens=4`** on the first
    four rows, the engine exhausts the budget part-way through the batch — later rows show **`budget_exhausted`**.

    Hydrate with a higher **`max_tokens`** budget completes the stragglers.
            """),
            mo.md("### After `ctx.map` (`max_tokens=4`)"),
            tok_run.select(
                "id",
                pl.col("ctx").ctx.estimate_tokens().alias("est_in"),
                pl.col("ai").ai.status().alias("status"),
            ),
            mo.md("### After hydrate (`max_tokens=20`)"),
            tok_hydrated.select(
                "id",
                pl.col("ai").ai.status().alias("status"),
            ),
        ]
    )
    return


@app.cell
def _(df, m_tuned, mo, pl):
    tuned = (
        df.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=m_tuned,
                max_concurrency=2,
                rate_limit_per_second=0,
            )
            .alias("ai"),
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
    ## 5. `max_concurrency` and `rate_limit_per_second`

    * **`max_concurrency`** — upper bound on concurrent fake-model tasks (still deterministic output).
    * **`rate_limit_per_second=0`** — disables the local governor (useful for fast `FakeModel` loops;
      restore a positive value when calling real providers).

    Here all rows succeed; the parameters still flow through to the Rust engine.
            """),
            tuned.select(
                "id",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.attempts().alias("attempts"),
            ),
        ]
    )
    return


@app.cell
def _(DEMO_CACHE, df, m_requests, mo, pl, shutil):
    shutil.rmtree(DEMO_CACHE, ignore_errors=True)
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)

    cached_first = (
        df.head(3)
        .lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=m_requests,
                cache=True,
                cache_path=str(DEMO_CACHE),
            )
            .alias("ai"),
        )
        .collect()
    )

    cached_second = (
        df.head(3)
        .lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=m_requests,
                max_requests=0,
                cache=True,
                cache_path=str(DEMO_CACHE),
            )
            .alias("ai"),
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
    ## 6. `cache` + `cache_path`

    Rows are written to **`cache_entries.jsonl`** after successful (or model-error) calls.

    A second **`ctx.map`** with **`max_requests=0`** cannot call the provider, but **disk hits** still return
    **`cache_hit`** with the stored value.
            """),
            mo.md("### First pass (cold cache)"),
            cached_first.select("id", pl.col("ai").ai.status().alias("status")),
            mo.md("### Second pass (`max_requests=0`, warm cache)"),
            cached_second.select("id", pl.col("ai").ai.status().alias("status")),
            mo.md(f"Cache directory: `{DEMO_CACHE}`"),
        ]
    )
    return


@app.cell
def _(DEMO_CACHE, df, m_requests, mo, pl, shutil):
    shutil.rmtree(DEMO_CACHE, ignore_errors=True)
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)

    partial = (
        df.head(4)
        .lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=m_requests,
                max_requests=1,
                cache=True,
                cache_path=str(DEMO_CACHE),
            )
            .alias("ai"),
        )
        .collect()
    )

    filled = (
        partial.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(
                ctx=pl.col("ctx"),
                model=m_requests,
                max_requests=5,
                cache=True,
                cache_path=str(DEMO_CACHE),
            )
            .alias("ai"),
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
    ## 7. Hydrate with cache enabled

    Hydration respects the same **`cache` / `cache_path` / budget`** knobs. New completions are appended to the JSONL
    store so later maps can **`cache_hit`** without spending **`max_requests`**.
            """),
            mo.md("### Partial map"),
            partial.select("id", pl.col("ai").ai.status().alias("status")),
            mo.md("### After hydrate (same cache dir)"),
            filled.select("id", pl.col("ai").ai.status().alias("status")),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Done

    You have exercised **`max_requests`**, **`max_tokens`**, **`max_concurrency`**, **`rate_limit_per_second`**,
    **`cache` / `cache_path`**, and **`.ai.hydrate(...)`** with matching parameters.

    See **`03_cache_back_to_back_and_hydrate.py`** for a narrative focused on cache reuse across successive frames.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
