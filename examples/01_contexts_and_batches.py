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
    # 01 - Context types and batches

    `polars-ai` treats model input as data. A single row-level input is an
    **`AiModelContext`**: a Polars `Struct` with fields for type, value, MIME
    type, and metadata. A grouped collection of contexts is a
    **`ContextBatch`**: a `List[AiModelContext]`.

    This notebook stays before model calls. The goal is to get comfortable
    creating, inspecting, grouping, and reducing context columns.
    """)
    return


@app.cell
def _(mo, pl):
    notes = pl.DataFrame(
        {
            "topic": ["shipping", "shipping", "shipping", "quality", "quality"],
            "position": [1, 2, 3, 1, 2],
            "note": [
                "Customer asked whether expedited delivery is available.",
                "Warehouse confirmed same-day pickup before 3pm.",
                "Carrier warned that rural routes may add one day.",
                "The first unit arrived with a cracked hinge.",
                "Replacement unit passed the visual inspection.",
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 1. Start with ordinary Polars data"),
            notes,
        ]
    )
    return (notes,)


@app.cell
def _(mo, notes, pl, pl_ai):
    notes_ctx = notes.with_columns(
        pl_ai.text_context(
            pl.col("note"),
            format_str="Support note: {value}",
        ).alias("ctx")
    )

    mo.vstack(
        [
            mo.md("## 2. Promote text into `AiModelContext`"),
            notes_ctx.select("topic", "position", "note", "ctx"),
        ]
    )
    return (notes_ctx,)


@app.cell
def _(mo, notes_ctx, pl_ai):
    ctx_dtype = notes_ctx["ctx"].dtype
    dtype_check = pl_ai.is_context_dtype(ctx_dtype)
    ctx_fields = [f"{field.name}: {field.dtype}" for field in ctx_dtype.fields]

    mo.vstack(
        [
            mo.md("## 3. Inspect the context dtype"),
            mo.md(f"`pl_ai.is_context_dtype(notes_ctx['ctx'].dtype)` -> `{dtype_check}`"),
            mo.md("Underlying struct fields:\n\n" + "\n".join(f"- `{field}`" for field in ctx_fields)),
        ]
    )
    return


@app.cell
def _(mo, notes_ctx, pl):
    inspected = notes_ctx.select(
        "topic",
        "position",
        pl.col("ctx").ctx.preview().alias("preview"),
        pl.col("ctx").ctx.estimate_tokens().alias("estimated_tokens"),
    )

    mo.vstack(
        [
            mo.md("## 4. Preview contexts and estimate tokens"),
            inspected,
        ]
    )
    return


@app.cell
def _(mo, notes_ctx, pl):
    topic_batches = (
        notes_ctx.lazy()
        .group_by("topic", maintain_order=True)
        .agg(pl.col("ctx").ctx.batch().alias("ctx_batch"))
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 5. Group context atoms into `ContextBatch` values"),
            topic_batches,
        ]
    )
    return (topic_batches,)


@app.cell
def _(mo, pl, pl_ai, topic_batches):
    batch_dtype = topic_batches["ctx_batch"].dtype
    batch_check = pl_ai.is_context_batch_dtype(batch_dtype)
    batch_inspection = topic_batches.select(
        "topic",
        pl.col("ctx_batch").ctxbatch.len().alias("batch_len"),
        pl.col("ctx_batch").ctxbatch.take(2).alias("first_two_contexts"),
    )

    mo.vstack(
        [
            mo.md("## 6. Inspect context batches"),
            mo.md(
                f"`pl_ai.is_context_batch_dtype(topic_batches['ctx_batch'].dtype)` -> `{batch_check}`"
            ),
            batch_inspection,
        ]
    )
    return


@app.cell
def _(mo, pl, topic_batches):
    reduced = topic_batches.with_columns(
        pl.col("ctx_batch")
        .ctxbatch.reduce_text(text_separator="\n\n", number_text_items=False)
        .alias("reduced_ctx")
    )

    reduced_display = reduced.select(
        "topic",
        pl.col("ctx_batch").ctxbatch.len().alias("source_contexts"),
        pl.col("reduced_ctx").ctx.preview().alias("reduced_preview"),
    )

    mo.vstack(
        [
            mo.md("""
            ## 7. Reduce a batch back to one text context

            `reduce_text` keeps grouped rows in order and returns another
            `AiModelContext`. That makes it usable anywhere a normal context
            column is expected.
            """),
            reduced_display,
        ]
    )
    return


@app.cell
def _(Path, base64):
    example_image_path = Path(__file__).parent / "static" / "example.jpg"
    example_image_b64 = base64.b64encode(example_image_path.read_bytes()).decode("utf-8")
    return example_image_b64, example_image_path


@app.cell
def _(example_image_b64, example_image_path, mo, pl, pl_ai):
    image_ctx = pl.DataFrame({"image_b64": [example_image_b64]}).with_columns(
        pl_ai.image_context(pl.col("image_b64"), mime="image/jpeg").alias("img_ctx")
    )

    mo.vstack(
        [
            mo.md("## 8. Image contexts are context atoms too"),
            mo.md(
                f"The example image lives at `{example_image_path.relative_to(example_image_path.parents[1])}`."
            ),
            image_ctx.select(
                pl.col("img_ctx").struct.field("_type").alias("type"),
                pl.col("img_ctx").struct.field("_mime").alias("mime"),
                pl.col("img_ctx").ctx.preview().str.slice(0, 80).alias("preview_head"),
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Done

    You have created context atoms, grouped them into context batches, and
    reduced batches back to text contexts.

    Next: open `examples/02_map_operations.py` to turn contexts into
    `AiResponse` structs with `.ctx.map(...)`.
    """)
    return


if __name__ == "__main__":
    app.run()
