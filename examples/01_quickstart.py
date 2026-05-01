import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai

    return mo, pl, pl_ai


@app.cell
def _(mo):
    mo.md("""
    # 01 - Quickstart

    `polars-ai` runs model inference over ordinary Polars expressions and
    returns an `AiResponse` struct with status, value, cache key, telemetry,
    error details, and timestamps.
    """)
    return


@app.cell
def _(mo, pl, pl_ai):
    reviews = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "review": [
                "Battery life is excellent and the case feels sturdy.",
                "The left earbud stopped pairing after two days.",
                "The blender handles frozen fruit without stalling.",
            ],
        }
    )
    model = pl_ai.FakeModel(
        prompt="Classify sentiment and explain briefly: {value}",
        tag="quickstart-v1",
    )

    result = reviews.with_columns(pl_ai.infer("review", model=model).alias("ai"))

    mo.vstack(
        [
            mo.md("## Run inference"),
            result.select("id", "review", "ai"),
        ]
    )
    return (result,)


@app.cell
def _(mo, pl, result):
    decoded = result.select(
        "id",
        pl.col("ai").struct.unnest()
    )

    mo.vstack([mo.md("## Read `AiResponse` fields"), decoded])
    return


@app.cell
def _(mo):
    # We want to add some markdown to instruct them to go to 02_budgets_cache.py
    mo.md("""
    ### This was an introduction to `polars-ai`. But there's more to explore!
    To see budgets and cache in action, check out `02_budgets_cache.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
