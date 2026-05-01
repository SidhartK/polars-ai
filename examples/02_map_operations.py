import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai
    from polars_ai import FakeModel

    return FakeModel, mo, pl, pl_ai


@app.cell
def _(mo):
    mo.md("""
    # 02 - Map operations

    A vanilla map turns context columns into **`AiResponse`** structs. Each row
    gets a status, value, deterministic cache key, telemetry, and error fields.

    This notebook focuses on `.ctx.map(...)` and `.ctxbatch.map(...)`. Hydration
    and durable disk caching come later.
    """)
    return


@app.cell
def _(mo, pl, pl_ai):
    reviews = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["audio", "audio", "kitchen", "kitchen", "travel"],
            "review": [
                "Battery life is excellent and the case feels sturdy.",
                "The left earbud stopped pairing after two days.",
                "The blender handles frozen fruit without stalling.",
                "The lid seal leaks when the jar is filled above halfway.",
                "The backpack is light, but the shoulder strap rubs.",
            ],
        }
    ).with_columns(
        pl_ai.text_context(
            pl.col("review"),
            format_str="Customer review: {value}",
        ).alias("ctx")
    )

    mo.vstack(
        [
            mo.md("## 1. Build a context column"),
            reviews.select("id", "category", "review", "ctx"),
        ]
    )
    return (reviews,)


@app.cell
def _(FakeModel):
    summarizer = FakeModel(prompt="Summarise the review in one phrase: {value}", tag="map-demo-v1")
    sentiment_model = FakeModel(prompt="Classify sentiment from this summary: {value}", tag="sentiment-v1")
    category_model = FakeModel(prompt="Summarise this group of reviews: {value}", tag="category-v1")
    return category_model, sentiment_model, summarizer


@app.cell
def _(mo, pl, reviews, summarizer):
    mapped = (
        reviews.lazy()
        .with_columns(pl.col("ctx").ctx.map(model=summarizer).alias("ai"))
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 2. `.ctx.map(...)` returns an `AiResponse` struct"),
            mapped.select("id", "review", "ai"),
        ]
    )
    return (mapped,)


@app.cell
def _(mapped, mo, pl):
    decoded = mapped.select(
        "id",
        pl.col("ai").ai.status().alias("status"),
        pl.col("ai").ai.value().alias("value"),
        pl.col("ai").ai.cache_key().str.slice(0, 16).alias("cache_key_head"),
        pl.col("ai").ai.attempts().alias("attempts"),
        pl.col("ai").ai.input_tokens().alias("input_tokens"),
        pl.col("ai").ai.output_tokens().alias("output_tokens"),
        pl.col("ai").ai.cost_usd().alias("cost_usd"),
    )

    mo.vstack(
        [
            mo.md("## 3. Unpack `AiResponse` with the `.ai` namespace"),
            decoded,
        ]
    )
    return


@app.cell
def _(mo, pl, reviews, summarizer):
    preflight_keys = reviews.select(
        "id",
        pl.col("ctx").ctx.cache_key(summarizer).str.slice(0, 24).alias("cache_key_head"),
    )

    mo.vstack(
        [
            mo.md("""
            ## 4. Compute cache keys without calling the model

            `ctx.cache_key(model)` fingerprints the context payload and the
            model config. It is useful for debugging, joins, or preflight checks.
            """),
            preflight_keys,
        ]
    )
    return


@app.cell
def _(mo, pl, reviews, summarizer):
    request_capped = (
        reviews.lazy()
        .with_columns(
            pl.col("ctx").ctx.map(model=summarizer, max_requests=2).alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 5. `max_requests` caps new completions

            Here only two rows are allowed to call the model. Later rows remain
            structured `AiResponse` values with status `budget_exhausted`.
            """),
            request_capped.select(
                "id",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
            ),
        ]
    )
    return


@app.cell
def _(FakeModel, mo, pl, pl_ai):
    token_examples = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "text": [
                "tiny",
                "a short sentence",
                "This longer review has enough characters to consume more of the token budget.",
                "This final row should be past a very small cumulative token budget.",
            ],
        }
    ).with_columns(
        pl_ai.text_context(pl.col("text"), format_str="Token demo: {value}").alias("ctx")
    )

    token_model = FakeModel(prompt="Process token demo: {value}", tag="token-budget-v1")
    token_capped = (
        token_examples.lazy()
        .with_columns(pl.col("ctx").ctx.map(model=token_model, max_tokens=8).alias("ai"))
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 6. `max_tokens` applies a rough cumulative input-token budget"),
            token_capped.select(
                "id",
                "text",
                pl.col("ctx").ctx.estimate_tokens().alias("estimated_tokens"),
                pl.col("ai").ai.status().alias("status"),
            ),
        ]
    )
    return


@app.cell
def _(mo, pl, reviews, summarizer):
    tuned = (
        reviews.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=summarizer,
                max_concurrency=2,
                rate_limit_per_second=0,
            )
            .alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 7. Concurrency and rate limit knobs

            `max_concurrency` and `rate_limit_per_second` control execution,
            not the shape of the response. `rate_limit_per_second=0` disables
            local pacing for this deterministic fake-model demo.
            """),
            tuned.select("id", pl.col("ai").ai.status().alias("status")),
        ]
    )
    return


@app.cell
def _(mo, pl, reviews, sentiment_model, summarizer):
    chained = (
        reviews.lazy()
        .with_columns(pl.col("ctx").ctx.map(model=summarizer).alias("summary_ai"))
        .with_columns(
            pl.col("summary_ai")
            .ai.to_context(format_str="Previous model answer: {value}")
            .alias("summary_ctx")
        )
        .with_columns(pl.col("summary_ctx").ctx.map(model=sentiment_model).alias("sentiment_ai"))
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 8. Chain maps by converting responses back to context

            A mapped column is an `AiResponse` struct, not a context. Use
            `.ai.to_context()` before feeding a previous answer to another map.
            """),
            chained.select(
                "id",
                pl.col("summary_ai").ai.value().alias("summary"),
                pl.col("sentiment_ai").ai.value().alias("sentiment"),
            ),
        ]
    )
    return


@app.cell
def _(category_model, mo, pl, reviews):
    category_batches = (
        reviews.lazy()
        .group_by("category", maintain_order=True)
        .agg(pl.col("ctx").ctx.batch().alias("ctx_batch"))
        .with_columns(
            pl.col("ctx_batch").ctxbatch.map(model=category_model).alias("category_ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 9. Batch maps call the model once per grouped batch

            `.ctx.batch()` collects row contexts per group. `.ctxbatch.map(...)`
            then invokes the model once for each `ContextBatch`.
            """),
            category_batches.select(
                "category",
                pl.col("ctx_batch").ctxbatch.len().alias("reviews_in_batch"),
                pl.col("category_ai").ai.status().alias("status"),
                pl.col("category_ai").ai.value().alias("value"),
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Done

    You have mapped contexts to `AiResponse` structs, inspected response
    fields, used budgets, chained model calls, and mapped grouped batches.

    Next: open `examples/03_hydration_real_dataset.py` to resume incomplete
    responses with `.ai.hydrate(...)`.
    """)
    return


if __name__ == "__main__":
    app.run()
