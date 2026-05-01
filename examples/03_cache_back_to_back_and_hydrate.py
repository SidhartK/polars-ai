"""Marimo notebook: disk cache across successive maps and hydrate."""

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
    # 03 — Cache across back-to-back calls + hydrate

    This notebook shows why **`cache=True`** matters when you:

    1. **Warm** the JSONL store with a first **`ctx.map`**.
    2. Run a **second map immediately** with **`max_requests=0`** — disk hits return **`cache_hit`**
       without spending provider budget.
    3. Add **new rows** that are not in the cache — they stay **`budget_exhausted`** until you **hydrate**
       with a positive **`max_requests`** while keeping the same **`cache_path`**.

    The demo directory **`__03_cache_demo__/`** sits beside this file (gitignored) and is reset when you run the setup cell.
    """)
    return


@app.cell
def _(Path, shutil):
    CACHE_DIR = Path(__file__).resolve().parent / "__03_cache_demo__"
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (CACHE_DIR,)


@app.cell
def _(CACHE_DIR, FakeModel, mo, pl, pl_ai):
    model = FakeModel(prompt="Label {value}", tag="cache-demo-03")

    warm_df = pl.DataFrame(
        {
            "run": ["warm"] * 4,
            "label": ["alpha", "beta", "gamma", "delta"],
            "payload": [
                "first-cache-payload-a",
                "second-cache-payload-b",
                "third-cache-payload-c",
                "fourth-cache-payload-d",
            ],
        }
    ).with_columns(pl_ai.text_context(pl.col("payload")).alias("ctx"))

    warm_result = (
        warm_df.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=model,
                cache=True,
                cache_path=str(CACHE_DIR),
            )
            .alias("ai"),
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
    ## 1. Warm the cache (first `ctx.map`)

    Four distinct payloads — each row performs a **`FakeModel`** call and appends to
    **`cache_entries.jsonl`** inside the cache directory.
            """),
            warm_result.select(
                "label",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().str.slice(0, 40).alias("value_head"),
            ),
        ]
    )
    return model, warm_df


@app.cell
def _(CACHE_DIR, mo, model, pl, warm_df):
    """Same contexts, zero request budget — disk replay only."""

    replay = (
        warm_df.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=model,
                max_requests=0,
                cache=True,
                cache_path=str(CACHE_DIR),
            )
            .alias("ai"),
        )
        .collect()
    )

    statuses = replay.select(pl.col("ai").ai.status()).to_series().to_list()

    mo.vstack(
        [
            mo.md("""
    ## 2. Back-to-back call with `max_requests=0`

    Because **`cache_hit`** short-circuits before the request budget is charged, every row can
    still resolve even when **`max_requests=0`** — values come straight from disk.
            """),
            replay.select(
                "label",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().str.slice(0, 40).alias("value_head"),
            ),
            mo.md(f"Observed statuses: `{statuses}`"),
        ]
    )
    return


@app.cell
def _(CACHE_DIR, mo, model, pl, pl_ai, warm_df):
    mixed = pl.concat(
        [
            warm_df,
            pl.DataFrame(
                {
                    "run": ["extend", "extend"],
                    "label": ["epsilon", "zeta"],
                    "payload": [
                        "brand-new-epsilon-payload",
                        "brand-new-zeta-payload",
                    ],
                }
            ).with_columns(pl_ai.text_context(pl.col("payload")).alias("ctx")),
        ],
        how="vertical",
    )

    cold_tail = (
        mixed.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=model,
                max_requests=0,
                cache=True,
                cache_path=str(CACHE_DIR),
            )
            .alias("ai"),
        )
        .collect()
    )

    tail_status = cold_tail.select(pl.col("ai").ai.status()).to_series().to_list()

    mo.vstack(
        [
            mo.md("""
    ## 3. Mixed frame — cached rows vs unseen rows

    We **concat** the original four rows (already on disk) with **two new payloads**.
    With **`max_requests=0`**, cached rows **`cache_hit`**, but unseen rows cannot call the model
    and remain **`budget_exhausted`**.
            """),
            cold_tail.select(
                "label",
                pl.col("ai").ai.status().alias("status"),
            ),
            mo.md(f"Status column: `{tail_status}`"),
        ]
    )
    return (cold_tail,)


@app.cell
def _(CACHE_DIR, cold_tail, mo, model, pl):
    hydrated = (
        cold_tail.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(
                ctx=pl.col("ctx"),
                model=model,
                max_requests=2,
                cache=True,
                cache_path=str(CACHE_DIR),
            )
            .alias("ai"),
        )
        .collect()
    )

    final_status = hydrated.select(pl.col("ai").ai.status()).to_series().to_list()

    mo.vstack(
        [
            mo.md("""
    ## 4. Hydrate with cache still enabled

    **`.ai.hydrate`** keeps **`cache_hit` / `ok`** rows intact and spends **`max_requests=2`** only on the
    two exhausted rows. Successful fills are appended to the same JSONL file for future runs.
            """),
            hydrated.select(
                "label",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().str.slice(0, 40).alias("value_head"),
            ),
            mo.md(f"Final statuses: `{final_status}`"),
        ]
    )
    return


@app.cell
def _(CACHE_DIR, mo):
    jsonl = CACHE_DIR / "cache_entries.jsonl"
    lines = 0
    if jsonl.is_file():
        lines = sum(1 for _ in jsonl.open(encoding="utf-8"))

    mo.vstack(
        [
            mo.md("## 5. On-disk artifact"),
            mo.md(
                f"**`{jsonl}`** — `{lines}` line(s). "
                "Each line is one cached row payload (append-only)."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Done

    **Takeaway:** treat the cache directory as a durable sidecar. Pair it with **hydration** when
    you need to resume partial batches without re-querying rows that already succeeded.

    See **`02_model_parameters_and_hydration.py`** for a full tour of every map / hydrate keyword.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
