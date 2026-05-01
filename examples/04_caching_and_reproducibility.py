import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import shutil
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars_ai as pl_ai
    from polars_ai import FakeModel

    return FakeModel, Path, json, mo, pl, pl_ai, shutil


@app.cell
def _(mo):
    mo.md("""
    # 04 - Caching and reproducibility

    Disk caching makes model results durable outside the lifetime of a
    dataframe object. A cache entry is keyed by the context payload and the
    model config, so the same inputs can be replayed later as `cache_hit`
    rows without spending request budget.
    """)
    return


@app.cell
def _(Path, shutil):
    cache_dir = Path(__file__).resolve().parent / "__04_cache_demo__"
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (cache_dir,)


@app.cell
def _(FakeModel, mo, pl, pl_ai):
    raw_records = [
        {"id": 101, "payload": "Extract the product name from: AeroPress filter pack"},
        {"id": 102, "payload": "Extract the product name from: insulated lunch tote"},
        {"id": 103, "payload": "Extract the product name from: ceramic pour-over dripper"},
        {"id": 104, "payload": "Extract the product name from: stainless travel mug"},
    ]

    original_df = pl.DataFrame(raw_records).with_columns(
        pl_ai.text_context(pl.col("payload"), format_str="{value}").alias("ctx")
    )
    cache_model = FakeModel(prompt="Return a compact product label: {value}", tag="cache-repro-v1")

    mo.vstack(
        [
            mo.md("## 1. Original dataframe and stable model config"),
            original_df.select("id", "payload", "ctx"),
        ]
    )
    return cache_model, original_df, raw_records


@app.cell
def _(cache_dir, cache_model, mo, original_df, pl):
    warm = (
        original_df.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=cache_model,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 2. Warm the cache"),
            warm.select(
                "id",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
                pl.col("ai").ai.cache_key().str.slice(0, 20).alias("cache_key_head"),
            ),
        ]
    )
    return (warm,)


@app.cell
def _(cache_dir, json, mo):
    jsonl_path = cache_dir / "cache_entries.jsonl"
    cache_lines = jsonl_path.read_text(encoding="utf-8").splitlines() if jsonl_path.is_file() else []
    first_entry = json.loads(cache_lines[0]) if cache_lines else {}
    first_entry_preview = {
        key: first_entry.get(key)
        for key in ["status", "cache_key", "value", "model_config"]
        if key in first_entry
    }

    mo.vstack(
        [
            mo.md("## 3. Inspect the on-disk sidecar"),
            mo.md(f"Cache file: `{jsonl_path}`"),
            mo.md(f"Cache entries: `{len(cache_lines)}`"),
            first_entry_preview,
        ]
    )
    return


@app.cell
def _(mo, original_df):
    discarded_original_df = None

    mo.md("""
    ## 4. Discard the original dataframe object

    For the rest of this notebook, we stop using `original_df` and keep only a
    placeholder named `discarded_original_df`. The cache is still on disk, and
    the raw records can be used to rebuild an equivalent frame.
    """)
    return (discarded_original_df,)


@app.cell
def _(cache_dir, cache_model, mo, pl, pl_ai, raw_records):
    rebuilt_df = pl.DataFrame(raw_records).with_columns(
        pl_ai.text_context(pl.col("payload"), format_str="{value}").alias("ctx")
    )

    replay = (
        rebuilt_df.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=cache_model,
                max_requests=0,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("## 5. Rebuild the dataframe and replay with zero request budget"),
            replay.select(
                "id",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
            ),
        ]
    )
    return rebuilt_df, replay


@app.cell
def _(cache_dir, cache_model, mo, pl, replay):
    reordered_replay = (
        replay.sort("id", descending=True)
        .lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=cache_model,
                max_requests=0,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 6. Row order does not matter

            Cache keys are based on context plus model config, not dataframe
            identity or row position.
            """),
            reordered_replay.select("id", pl.col("ai").ai.status().alias("status")),
        ]
    )
    return


@app.cell
def _(cache_dir, cache_model, mo, pl, pl_ai, raw_records):
    changed_records = [
        dict(record)
        for record in raw_records
    ]
    changed_records[-1] = {
        "id": changed_records[-1]["id"],
        "payload": "Extract the product name from: leakproof camping thermos",
    }

    changed_df = pl.DataFrame(changed_records).with_columns(
        pl_ai.text_context(pl.col("payload"), format_str="{value}").alias("ctx")
    )

    mixed = (
        changed_df.lazy()
        .with_columns(
            pl.col("ctx")
            .ctx.map(
                model=cache_model,
                max_requests=0,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 7. Changed inputs create new cache keys

            The first three rows match the warmed cache. The changed payload has
            no cache entry and cannot call the model because `max_requests=0`.
            """),
            mixed.select(
                "id",
                "payload",
                pl.col("ai").ai.status().alias("status"),
            ),
        ]
    )
    return (mixed,)


@app.cell
def _(cache_dir, cache_model, mixed, mo, pl):
    filled = (
        mixed.lazy()
        .with_columns(
            pl.col("ai")
            .ai.hydrate(
                ctx=pl.col("ctx"),
                model=cache_model,
                max_requests=1,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("ai")
        )
        .collect()
    )

    mo.vstack(
        [
            mo.md("""
            ## 8. Hydrate the one unseen row

            Hydration preserves the cache hits and spends one request on the
            changed row. The new success is appended to the same cache file.
            """),
            filled.select(
                "id",
                pl.col("ai").ai.status().alias("status"),
                pl.col("ai").ai.value().alias("value"),
            ),
        ]
    )
    return (filled,)


@app.cell
def _(cache_dir, mo):
    jsonl_path = cache_dir / "cache_entries.jsonl"
    final_lines = jsonl_path.read_text(encoding="utf-8").splitlines() if jsonl_path.is_file() else []

    mo.vstack(
        [
            mo.md("## 9. Final cache size"),
            mo.md(f"`{jsonl_path}` now has `{len(final_lines)}` entries."),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Done

    Same context payload + same model config + same cache path gives replayable
    `cache_hit` rows. Changing either the context or model config creates a new
    key, which can be filled later with hydration.
    """)
    return


if __name__ == "__main__":
    app.run()
