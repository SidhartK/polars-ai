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

    This notebook walks through the core `polars_ai` flow: create model contexts,
    inspect them, map a fake model over them, and optionally call OpenAI if
    `OPENAI_API_KEY` is set.
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
def _(df_ctx, pl, summariser):
    result = df_ctx.lazy().limit(2).select("id", pl.col("ctx").ctx.map(model=summariser).alias("ai_output"))

    print(result.explain(optimized=True))
    return


@app.cell
def _(df_ctx, mo, pl, summariser):
    result = (
        df_ctx.lazy()
        .with_columns(pl.col("ctx").ctx.map(model=summariser).alias("ai_output"))
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 4. `ctx.map()` with `FakeModel`"),
            result.select("id", "review", "ai_output"),
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
        .with_columns(pl.col("ctx").ctx.map(model=step1).alias("summary"))
        .with_columns(pl_ai.text_context(pl.col("summary")).alias("summary_ctx"))
        .with_columns(pl.col("summary_ctx").ctx.map(model=step2).alias("sentiment"))
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 5. Chained `ctx.map()` calls"),
            chained.select("id", "review", "summary", "sentiment"),
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
            mo.md("## 6. Image context struct shape"),
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
            df_ctx.head(2)
            .lazy()
            .with_columns(
                pl.col("ctx").ctx.map(model=openai_text_model).alias("openai_summary")
            )
            .collect()
        )
        openai_images = (
            df_img.head(2)
            .lazy()
            .with_columns(
                pl.col("img_ctx")
                .ctx.map(model=openai_image_model)
                .alias("openai_image_description")
            )
            .collect()
        )
        openai_output = mo.vstack(
            [
                mo.md("## 7. OpenAI examples"),
                mo.md("### Text input"),
                openai_text.select("id", "review", "openai_summary"),
                mo.md("### Base64 image input"),
                openai_images.select("id", "openai_image_description"),
            ]
        )
    else:
        openai_text = None
        openai_images = None
        openai_output = mo.md(
            "## 7. OpenAI examples\n\nSkipping because `OPENAI_API_KEY` is not set."
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
            mo.md("## 8. `pl_ai.context()` convenience wrapper"),
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
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
