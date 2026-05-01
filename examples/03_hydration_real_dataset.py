import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import urllib.request
    import zipfile
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai
    from polars_ai import FakeModel

    return FakeModel, Path, io, mo, pl, pl_ai, urllib, zipfile


@app.cell
def _(mo):
    mo.md("""
    # 03 - Hydration on a real text dataset

    Hydration completes rows that already have an `AiResponse` but are not
    finished yet. The usual reason is a budget: a first run processes part of a
    frame, marks the rest `budget_exhausted`, and a later run resumes from the
    saved response column.

    This notebook uses a small sample from the UCI SMS Spam Collection.
    """)
    return


@app.cell
def _(Path):
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    data_dir = Path(__file__).resolve().parent / "__03_data__"
    sms_path = data_dir / "SMSSpamCollection"
    return data_dir, data_url, sms_path


@app.cell
def _(data_dir, data_url, io, mo, sms_path, urllib, zipfile):
    if not sms_path.is_file():
        data_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(data_url) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extract("SMSSpamCollection", path=data_dir)

    mo.md(f"Dataset file: `{sms_path}`")
    return


@app.cell
def _(mo, pl, sms_path):
    sms_raw = pl.read_csv(
        sms_path,
        separator="\t",
        has_header=False,
        new_columns=["label", "message"],
    ).with_row_index("row_id")

    sms_sample = pl.concat(
        [
            sms_raw.filter(pl.col("label") == "ham").head(12),
            sms_raw.filter(pl.col("label") == "spam").head(12),
        ],
        how="vertical",
    ).sort("row_id")

    mo.vstack(
        [
            mo.md("## 1. Load a small balanced sample"),
            sms_sample.select("row_id", "label", "message"),
        ]
    )
    return (sms_sample,)


@app.cell
def _(mo, pl, pl_ai, sms_sample):
    sms_ctx = sms_sample.with_columns(
        pl_ai.text_context(
            pl.col("message"),
            format_str="Classify this SMS message as ham or spam, then explain briefly: {value}",
        ).alias("ctx")
    )

    mo.vstack(
        [
            mo.md("## 2. Create classification contexts"),
            sms_ctx.select(
                "row_id",
                "label",
                pl.col("ctx").ctx.preview().str.slice(0, 120).alias("context_preview"),
            ),
        ]
    )
    return (sms_ctx,)


@app.cell
def _(FakeModel):
    sms_model = FakeModel(
        prompt="Classify as ham or spam, then explain briefly: {value}",
        tag="sms-hydration-v1",
    )
    return (sms_model,)


@app.cell
def _(mo, pl, sms_ctx, sms_model):
    partial = (
        sms_ctx.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(model=sms_model, max_requests=8)
            .alias("ai")
        )
        .collect()
    )

    partial_counts = (
        partial.select(pl.col("ai").ai.status().alias("status"))
        .group_by("status")
        .len()
        .sort("status")
    )

    mo.vstack(
        [
            mo.md("""
            ## 3. First pass: stop partway through the dataset

            `max_requests=8` lets the first eight model calls complete. The
            remaining rows keep their context and response shape, but their
            status is `budget_exhausted`.
            """),
            partial_counts,
            partial.select(
                "row_id",
                "label",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
            ),
        ]
    )
    return (partial,)


@app.cell
def _(mo, partial, pl):
    preserved = partial.select(
        "row_id",
        "label",
        pl.col("ctx").ctx.preview().str.slice(0, 80).alias("ctx_preview"),
        pl.col("ai").ai.status().alias("status"),
        pl.col("ai").ai.cache_key().str.slice(0, 20).alias("cache_key_head"),
        pl.col("ai").ai.value().alias("value"),
    )

    mo.vstack(
        [
            mo.md("""
            ## 4. A partial result is still useful state

            Hydration needs two columns from this frame: the original `ctx` and
            the previous `ai` response. Completed rows already have values;
            exhausted rows carry enough structure to be resumed.
            """),
            preserved,
        ]
    )
    return


@app.cell
def _(mo, partial, pl, sms_model):
    hydrated = (
        partial.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(ctx=pl.col("ctx"), model=sms_model, max_requests=32)
            .alias("ai")
        )
        .collect()
    )

    before_counts = (
        partial.select(pl.col("ai").ai.status().alias("status"))
        .group_by("status")
        .len()
        .sort("status")
        .rename({"len": "before"})
    )
    after_counts = (
        hydrated.select(pl.col("ai").ai.status().alias("status"))
        .group_by("status")
        .len()
        .sort("status")
        .rename({"len": "after"})
    )
    status_compare = before_counts.join(after_counts, on="status", how="full", coalesce=True)

    mo.vstack(
        [
            mo.md("""
            ## 5. Hydrate with a fresh budget

            `.ai.hydrate(...)` preserves terminal rows and spends the new
            budget only on incomplete rows.
            """),
            status_compare,
        ]
    )
    return (hydrated,)


@app.cell
def _(hydrated, mo, partial, pl):
    exhausted_before = partial.filter(
        pl.col("ai").ai.status() == "budget_exhausted"
    ).select(
        "row_id",
        "label",
        "message",
        pl.col("ai").ai.status().alias("before_status"),
    )

    completed_after = hydrated.join(
        exhausted_before.select("row_id"),
        on="row_id",
        how="inner",
    ).select(
        "row_id",
        "label",
        "message",
        pl.col("ai").ai.status().alias("after_status"),
        pl.col("ai").ai.value().alias("after_value"),
    )

    mo.vstack(
        [
            mo.md("## 6. Inspect the rows that hydration filled"),
            mo.md("### Before hydration"),
            exhausted_before,
            mo.md("### After hydration"),
            completed_after,
        ]
    )
    return


@app.cell
def _(hydrated, mo, pl, sms_model):
    no_op = (
        hydrated.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(ctx=pl.col("ctx"), model=sms_model, max_requests=0)
            .alias("ai")
        )
        .collect()
    )

    no_op_counts = (
        no_op.select(pl.col("ai").ai.status().alias("status"))
        .group_by("status")
        .len()
        .sort("status")
    )

    mo.vstack(
        [
            mo.md("""
            ## 7. Hydrating a complete frame is a no-op

            Once every row is terminal, `max_requests=0` is enough because
            there are no incomplete rows left to call.
            """),
            no_op_counts,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Done

    You used a real text dataset to create a partial AI result, preserved the
    original context and response columns, and hydrated the incomplete rows.

    Next: open `examples/04_caching_and_reproducibility.py` to replay results
    from disk after rebuilding the dataframe.
    """)
    return


if __name__ == "__main__":
    app.run()
