import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import json
    import os
    from dataclasses import dataclass
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai
    from polars_ai import AiModel, FakeModel

    return AiModel, FakeModel, Path, base64, dataclass, json, mo, os, pl, pl_ai


@app.cell
def _(mo):
    mo.md("""
    # polars-ai quickstart

    This notebook walks through the durable AI flow: **`AiModelContext`**
    structs from `text_context()`, **`.ctx.map()`** yielding **`AiResponse`**
    (status/value/cache key), budgets, **`.ai.hydrate()`** for partial runs, and
    optional OpenAI cells when **`OPENAI_API_KEY`** is set.

    Mapped columns are structs — unwrap text with **`pl.col("ai").ai.value()`** before
    feeding a second-stage model (`text_context` cannot take a struct column).
    """)
    return


@app.cell
def _(Path, base64):
    EXAMPLE_IMAGE_PATH = Path(__file__).parent / "static" / "example.jpg"
    EXAMPLE_JPG_B64 = base64.b64encode(EXAMPLE_IMAGE_PATH.read_bytes()).decode("utf-8")
    return EXAMPLE_IMAGE_PATH, EXAMPLE_JPG_B64


@app.cell
def _(AiModel, dataclass, json):
    @dataclass
    class OpenAIModel(AiModel):
        """
        OpenAI-backed model config for `.ctx.map()`.

        The Rust plugin reads this JSON config, combines it with each row's
        AiModelContext, and calls OpenAI using `OPENAI_API_KEY`.
        """

        prompt: str
        model: str = "gpt-4o-mini"
        temperature: float = 0.0
        max_tokens: int = 128

        @property
        def model_config(self) -> str:
            return json.dumps(
                {
                    "provider": "openai",
                    "prompt": self.prompt,
                    "model": self.model,
                    "options": {
                        "max_output_tokens": self.max_tokens,
                    },
                }
            )

    return (OpenAIModel,)


@app.cell
def _(EXAMPLE_JPG_B64, mo, pl):
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "review": [
                "This product is absolutely fantastic, would buy again.",
                "Terrible quality. Broke after one day.",
                "Decent value for the price, nothing special.",
                "Best purchase I have made this year!",
                "Would not recommend to anyone.",
            ],
            "image_b64": [EXAMPLE_JPG_B64] * 5,
        }
    )

    mo.vstack([mo.md("## 1. Original DataFrame"), df])
    return (df,)


@app.cell
def _(df, mo, pl, pl_ai):
    df_ctx = df.with_columns(
        pl_ai.text_context(pl.col("review"), format_str="Review: {value}").alias("ctx"),
    )

    mo.vstack(
        [
            mo.md("## 2. With context column"),
            df_ctx.select("id", "ctx"),
            mo.md(f"`ctx` dtype: `{df_ctx['ctx'].dtype}`"),
        ]
    )
    return (df_ctx,)


@app.cell
def _(df_ctx, mo, pl_ai):
    assert pl_ai.is_context_dtype(df_ctx["ctx"].dtype), "dtype check failed"

    mo.md("`pl_ai.is_context_dtype(df_ctx['ctx'].dtype)` returns `True`.")
    return


@app.cell
def _(df_ctx, mo, pl):
    df_inspect = df_ctx.with_columns(
        pl.col("ctx").ctx.preview().alias("preview"),
        pl.col("ctx").ctx.estimate_tokens().alias("estimated_tokens"),
    )

    mo.vstack(
        [
            mo.md("## 3. Preview and token estimate"),
            df_inspect.select("id", "preview", "estimated_tokens"),
        ]
    )
    return


@app.cell
def _(FakeModel):
    summariser = FakeModel(prompt="Summarise this review: {value}", tag="summariser-v1")
    return (summariser,)


@app.cell
def _(df_ctx, mo, pl, summariser):
    mapped = (
        df_ctx.lazy()
        .with_columns(pl.col("ctx").ctx.map(model=summariser).alias("ai"))
        .collect()
    )

    unpacked = mapped.select(
        "id",
        "review",
        pl.col("ai").ai.status().alias("status"),
        pl.col("ai").ai.value().alias("summary_text"),
        pl.col("ai").ai.cache_key().alias("cache_key_preview"),
    )

    mo.vstack(
        [
            mo.md("""
    ## 4. `ctx.map()` returns `AiResponse`

    Each row under `ai` is an **`AiResponse`** struct (`status`, `value`, caching
    fields, timestamps). Inspect it with **`pl.col(\"ai\").ai.*`** — here we pull
    human-readable **`summary_text`** and the row **`status`** (`ok`,
    `budget_exhausted`, `cache_hit`, …).
            """),
            unpacked,
            mo.md(
                "`pl_ai.DEFAULT_CACHE_FOLDER` is `__polars_ai_cache__` when disk "
                "caching is enabled via `cache=True` on `.ctx.map` / `.ai.hydrate`."
            ),
        ]
    )
    return


@app.cell
def _(df_ctx, mo, pl, summariser):
    capped = (
        df_ctx.head(5)
        .lazy()
        .with_columns(pl.col("ctx").ctx.map(model=summariser, max_requests=2).alias("ai"))
        .collect()
    )

    statuses = capped.select(pl.col("ai").ai.status()).to_series().to_list()

    mo.vstack(
        [
            mo.md("""
    ## 5. Budgets (`max_requests`, `max_tokens`, …)

    `max_requests` caps provider completions across the frame (after cache hits row
    budget is not reused). Rows past the threshold keep struct columns with status
    **`budget_exhausted`** so you can filter or hydrate later.

    This slice has five reviews but only **`max_requests=2`** succeeds in order.
    Statuses: `{}`.
            """.format(statuses)),
            capped.select(
                "id",
                "review",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
            ),
        ]
    )
    return (capped,)


@app.cell
def _(capped, mo, pl, summariser):
    hydrated = (
        capped.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(ctx=pl.col("ctx"), model=summariser, max_requests=10)
            .alias("ai")
        )
        .collect()
    )

    final_statuses = hydrated.select(pl.col("ai").ai.status()).to_series().to_list()

    mo.vstack(
        [
            mo.md("""
    ## 6. Hydration (`pl.col(\"ai\").ai.hydrate`)

    Reattach the **`ctx`** column and call **`.hydrate`** with a fresh budget. Rows
    already terminal (`ok`, `cache_hit`, …) are left unchanged.

    After raising the budget (`max_requests=10`), statuses are `{}`.
            """.format(final_statuses)),
            hydrated.select(
                "id",
                "review",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
            ),
        ]
    )
    return


@app.cell
def _(FakeModel):
    step1 = FakeModel(prompt="Summarise: {value}", tag="step-1-summary")
    step2 = FakeModel(prompt="Classify sentiment: {value}", tag="step-2-sentiment")
    return step1, step2


@app.cell
def _(df, mo, pl, pl_ai, step1, step2):
    chained = (
        df.lazy()
        .with_columns(pl_ai.text_context(pl.col("review")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=step1).alias("summary_struct"))
        .with_columns(
            pl_ai.text_context(pl.col("summary_struct").ai.value()).alias("summary_ctx")
        )
        .with_columns(pl.col("summary_ctx").ctx.map(model=step2).alias("sentiment_struct"))
        .collect()
    )

    chained_display = chained.select(
        "id",
        "review",
        pl.col("summary_struct").ai.value().alias("summary"),
        pl.col("sentiment_struct").ai.value().alias("sentiment"),
    )

    mo.vstack(
        [
            mo.md("""
    ## 7. Chained `ctx.map()` calls

    Intermediate columns are structs. Build the next **`text_context`** from
    **`pl.col(\"summary_struct\").ai.value()`**, not from the bare struct column.
            """),
            chained_display,
        ]
    )
    return


@app.cell
def _(df, mo, pl, pl_ai):
    df_img = df.with_columns(
        pl_ai.image_context(pl.col("image_b64"), mime="image/jpeg").alias("img_ctx")
    )

    mo.vstack(
        [
            mo.md("## 8. Image context struct shape"),
            df_img.select("img_ctx").head(2),
            mo.md(f"`img_ctx` dtype: `{df_img['img_ctx'].dtype}`"),
        ]
    )
    return (df_img,)


@app.cell
def _(OpenAIModel, df_ctx, df_img, mo, os, pl):
    if os.getenv("OPENAI_API_KEY"):
        openai_text_model = OpenAIModel(
            prompt="Summarise this review in five words or fewer: {value}",
            max_tokens=32,
        )
        openai_image_model = OpenAIModel(
            prompt="Describe the main subject of this image in one short sentence.",
            max_tokens=64,
        )

        openai_text = (
            df_ctx.lazy()
            .limit(2)
            .with_columns(
                pl.col("ctx").ctx.map(model=openai_text_model).alias("openai_summary")
            )
            .collect()
        )
        openai_images = (
            df_img.lazy()
            .limit(2)
            .with_columns(
                pl.col("img_ctx")
                .ctx.map(model=openai_image_model)
                .alias("openai_image_description")
            )
            .collect()
        )
        openai_output = mo.vstack(
            [
                mo.md("## 9. OpenAI examples"),
                mo.md("### Text input"),
                openai_text.select(
                    "id",
                    "review",
                    pl.col("openai_summary").ai.value().alias("summary_text"),
                    pl.col("openai_summary").ai.status().alias("status"),
                ),
                mo.md("### Base64 image input"),
                openai_images.select(
                    "id",
                    pl.col("openai_image_description").ai.value().alias("caption"),
                    pl.col("openai_image_description").ai.status().alias("status"),
                ),
            ]
        )
    else:
        openai_text = None
        openai_images = None
        openai_output = mo.md(
            "## 9. OpenAI examples\n\nSkipping because `OPENAI_API_KEY` is not set."
        )

    openai_output
    return


@app.cell
def _(df, mo, pl, pl_ai):
    df_mixed = df.with_columns(
        pl_ai.context(pl.col("review"), kind="text", format_str="Analyse: {value}").alias(
            "ctx2"
        )
    )

    mo.vstack(
        [
            mo.md("## 10. `pl_ai.context()` convenience wrapper"),
            df_mixed.select("ctx2").head(2),
        ]
    )
    return


@app.cell
def _(EXAMPLE_IMAGE_PATH, mo):
    mo.md(f"""
    ## Done

    All quickstart cells completed. The image example uses
    `{EXAMPLE_IMAGE_PATH.relative_to(EXAMPLE_IMAGE_PATH.parents[2])}`.

    **Next:** open `examples/02_model_parameters_and_hydration.py` for a walkthrough of every
    `.ctx.map` / `.ai.hydrate` parameter (budgets, concurrency, rate limit, cache) and more hydration patterns
    (`marimo edit examples/02_model_parameters_and_hydration.py`).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
