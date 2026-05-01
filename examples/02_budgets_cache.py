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

    return Path, mo, pl, pl_ai, shutil


@app.cell
def _(mo):
    mo.md("""
    # 02 - Budgets and cache

    Budgets cap new provider calls. A disk cache makes completed rows replayable
    later without spending request budget.
    """)
    return


@app.cell
def _(Path, shutil):
    cache_dir = Path(__file__).resolve().parent / "__02_cache__"
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@app.cell
def _(cache_dir, mo, pl, pl_ai):
    model = pl_ai.FakeModel(prompt="Extract a short product label: {value}", tag="cache-v1")
    records = pl.DataFrame(
        {
            "payload": [
                "AeroPress filter pack",
                "insulated lunch tote",
                "ceramic pour-over dripper",
                "stainless travel mug",
            ]
        }
    )

    partial = records.with_columns(
        pl_ai.infer(
            "payload",
            model=model,
            cache_path=str(cache_dir),
            max_requests=2,
        ).alias("ai")
    )

    mo.vstack(
        [
            mo.md("## Cap new calls with `max_requests=2`"),
            partial.select(
                "payload",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("value").alias("value"),
            ),
        ]
    )
    return cache_dir, model, records


@app.cell
def _(cache_dir, model, mo, pl, pl_ai, records):
    replay = records.with_columns(
        pl_ai.infer(
            "payload",
            model=model,
            cache_path=str(cache_dir),
            max_requests=0,
        ).alias("ai")
    )

    mo.vstack(
        [
            mo.md("## Replay completed rows from disk with zero budget"),
            replay.select(
                "payload",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("cache_key").str.slice(0, 18).alias("cache_key"),
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
