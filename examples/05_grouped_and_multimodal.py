import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai

    return Path, base64, mo, pl, pl_ai


@app.cell
def _(mo):
    mo.md("""
    # 05 - Grouped and multimodal inputs

    `infer(...)` accepts ordinary strings, grouped `List[String]` expressions,
    and image-like inputs selected with `input_type`.
    """)
    return


@app.cell
def _(mo, pl, pl_ai):
    notes = pl.DataFrame(
        {
            "topic": ["shipping", "shipping", "quality"],
            "note": [
                "Customer asked about expedited delivery.",
                "Warehouse confirmed same-day pickup.",
                "Replacement unit passed inspection.",
            ],
        }
    )
    model = pl_ai.FakeModel(prompt="Summarize these notes:\n{value}", tag="group-v1")
    grouped = notes.group_by("topic", maintain_order=True).agg(
        pl_ai.infer(
            pl.col("note").implode(),
            model=model,
            text_separator="\n",
            number_text_items=True,
        ).alias("ai")
    )

    mo.vstack(
        [
            mo.md("## Group rows with normal Polars list expressions"),
            grouped.select(
                "topic",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("value").alias("value"),
            ),
        ]
    )
    return


@app.cell
def _(Path, base64, mo, pl, pl_ai):
    image_path = Path(__file__).parent / "static" / "example.jpg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_df = pl.DataFrame({"image_b64": [image_b64]})
    model = pl_ai.FakeModel(prompt="Describe this image: {value}", tag="image-demo")
    image_result = image_df.with_columns(
        pl_ai.infer(
            "image_b64",
            model=model,
            input_type="image",
            mime="image/jpeg",
        ).alias("ai")
    )

    mo.vstack(
        [
            mo.md("## Image inputs use `input_type`"),
            image_result.select(
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("error").alias("error"),
            ),
            mo.md("`FakeModel` intentionally reports a provider error for image inputs."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
