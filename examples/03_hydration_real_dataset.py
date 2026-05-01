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
    # 03 - Hydration

    Hydration completes rows that already have an `AiResponse` but stopped with
    an incomplete status, usually because an earlier run hit a budget.
    """)
    return


@app.cell
def _(mo, pl, pl_ai):
    sms = pl.DataFrame(
        {
            "row_id": list(range(1, 9)),
            "message": [
                "I'll call you when I arrive",
                "WIN a free prize now",
                "Can you pick up milk?",
                "URGENT claim your reward",
                "Meeting moved to 3pm",
                "Free entry in our draw",
                "Dinner at seven?",
                "Text STOP to unsubscribe",
            ],
        }
    )
    model = pl_ai.FakeModel(
        prompt="Classify this SMS as ham or spam and explain briefly: {value}",
        tag="sms-hydration-v1",
    )
    partial = sms.with_columns(
        pl_ai.infer("message", model=model, max_requests=3).alias("ai")
    )

    mo.vstack(
        [
            mo.md("## First pass: stop partway through"),
            partial.select(
                "row_id",
                "message",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("value").alias("value"),
            ),
        ]
    )
    return model, partial


@app.cell
def _(mo, model, partial, pl, pl_ai):
    hydrated = partial.with_columns(
        pl_ai.hydrate(
            response="ai",
            input="message",
            model=model,
            max_requests=3,
        ).alias("ai")
    )

    mo.vstack(
        [
            mo.md("## Hydrate with a fresh budget"),
            hydrated.select(
                "row_id",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("value").alias("value"),
            ),
        ]
    )
    return (hydrated,)


@app.cell
def _(hydrated, model, pl_ai):
    hydrated.with_columns(pl_ai.hydrate(response="ai", input="message", model=model, max_requests=3).alias("ai"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
