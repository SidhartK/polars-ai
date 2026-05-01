"""Integration tests for ContextBatch reduce/map flows (requires compiled extension)."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def _values(series: pl.Series) -> list[str | None]:
    return (
        series.to_frame("out")
        .select(pl.col("out").ai.value().alias("v"))
        .get_column("v")
        .to_list()
    )


def test_ctxbatch_reduce_text_order_and_map_fake_model() -> None:
    model = pl_ai.FakeModel(prompt="Echo:\n{value}", tag="reduce_two")
    df = (
        pl.DataFrame({"topic": ["A", "A", "B"], "msg": ["x", "y", "z"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("topic", maintain_order=True)
        .agg(
            pl.col("ctx")
            .ctx.batch()
            .ctxbatch.reduce_text()
            .ctx.map(model=model)
            .alias("out")
        )
        .sort("topic")
    )

    assert df.height == 2
    assert pl_ai.is_response_dtype(df["out"].dtype)
    statuses = df.select(pl.col("out").ai.status()).to_series().to_list()
    assert statuses == [pl_ai.RESPONSE_STATUS_OK, pl_ai.RESPONSE_STATUS_OK]

    vals = _values(df["out"])
    assert vals[0] is not None and vals[1] is not None
    assert "[FAKE | tag=reduce_two | type=text |" in vals[0]
    m0 = re.search(r"input_len=(\d+)", vals[0])
    m1 = re.search(r"input_len=(\d+)", vals[1])
    assert m0 and m1
    assert int(m0.group(1)) > int(m1.group(1))
    assert "x" in vals[0] and "y" in vals[0]
    assert "z" in vals[1]


def test_ctx_reduce_compatibility_wrapper() -> None:
    model = pl_ai.FakeModel(prompt="ORDER:{value}", tag="ord")
    df = (
        pl.DataFrame({"g": ["g"] * 3, "step": [0, 1, 2], "msg": ["first", "second", "third"]})
        .sort("step")
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("g", maintain_order=True)
        .agg(pl.col("ctx").ctx.reduce(model=model).alias("out"))
    )
    v = df.select(pl.col("out").ai.value()).to_series().item()
    assert v is not None
    assert df.select(pl.col("out").ai.status()).to_series().item() == pl_ai.RESPONSE_STATUS_OK
    assert "first\n\nsecond\n\nthird" in v


def test_ctx_reduce_text_separator_changes_cache_key() -> None:
    model = pl_ai.FakeModel(prompt="{value}", tag="sep")
    base = (
        pl.DataFrame({"topic": ["t", "t"], "msg": ["a", "b"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("topic", maintain_order=True)
    )
    k1 = (
        base.agg(pl.col("ctx").ctx.reduce(model=model, text_separator="\n").alias("out"))
        .select(pl.col("out").ai.cache_key())
        .item()
    )
    k2 = (
        base.agg(pl.col("ctx").ctx.reduce(model=model, text_separator="||").alias("out"))
        .select(pl.col("out").ai.cache_key())
        .item()
    )
    assert k1 != k2


def test_ctx_reduce_number_text_items_style() -> None:
    model = pl_ai.FakeModel(prompt="P:{value}", tag="num")
    df = (
        pl.DataFrame({"g": ["g", "g"], "msg": ["one", "two"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("g", maintain_order=True)
        .agg(
            pl.col("ctx")
            .ctx.reduce(model=model, number_text_items=True, text_separator="\n")
            .alias("out")
        )
    )
    v = df.select(pl.col("out").ai.value()).to_series().item()
    assert v is not None
    assert "Item 1:" in v and "Item 2:" in v


def test_ctxbatch_map_fake_provider_text_batch() -> None:
    model = pl_ai.FakeModel(prompt="Batch:{value}", tag="batch")
    df = (
        pl.DataFrame({"g": ["g", "g"], "msg": ["one", "two"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("g", maintain_order=True)
        .agg(pl.col("ctx").ctx.batch().ctxbatch.map(model=model).alias("out"))
    )
    assert df.select(pl.col("out").ai.status()).to_series().item() == pl_ai.RESPONSE_STATUS_OK
    assert "one" in df.select(pl.col("out").ai.value()).to_series().item()


def test_ctxbatch_reduce_image_fake_model_error() -> None:
    model = pl_ai.FakeModel(prompt="{value}", tag="img")
    tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    df = (
        pl.DataFrame({"g": ["g"], "buf": [tiny_b64]})
        .with_columns(pl_ai.image_context(pl.col("buf"), mime="image/png").alias("ctx"))
        .group_by("g")
        .agg(pl.col("ctx").ctx.reduce(model=model).alias("out"))
    )
    assert (
        df.select(pl.col("out").ai.status()).to_series().item()
        == pl_ai.RESPONSE_STATUS_MODEL_ERROR
    )

def test_ctx_reduce_max_requests_zero_budget_exhausted_all_groups() -> None:
    model = pl_ai.FakeModel()
    df = (
        pl.DataFrame({"topic": ["a", "b"], "msg": ["1", "2"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("topic", maintain_order=True)
        .agg(
            pl.col("ctx").ctx.reduce(model=model, max_requests=0).alias("out"),
        )
        .sort("topic")
    )
    stats = df.select(pl.col("out").ai.status()).to_series().to_list()
    assert stats == [pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED] * 2


def test_ctx_reduce_disk_cache_hit_second_collect(tmp_path) -> None:
    cache_dir = tmp_path / "reduce_cache"
    model = pl_ai.FakeModel(prompt="Reduce {value}", tag="rc")

    base = (
        pl.LazyFrame({"topic": ["t"], "msg": ["cached"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("topic")
        .agg(
            pl.col("ctx")
            .ctx.reduce(
                model=model,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("out")
        )
    )

    df1 = base.collect()
    assert df1.select(pl.col("out").ai.status()).to_series().item() == pl_ai.RESPONSE_STATUS_OK
    v1 = df1.select(pl.col("out").ai.value()).to_series().item()

    df2 = (
        pl.LazyFrame({"topic": ["t"], "msg": ["cached"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("topic")
        .agg(
            pl.col("ctx")
            .ctx.reduce(
                model=model,
                max_requests=0,
                cache=True,
                cache_path=str(cache_dir),
            )
            .alias("out")
        )
        .collect()
    )
    assert df2.select(pl.col("out").ai.status()).to_series().item() == pl_ai.RESPONSE_STATUS_CACHE_HIT
    assert df2.select(pl.col("out").ai.value()).to_series().item() == v1


def test_ctx_reduce_null_value_in_group_invalid_context() -> None:
    model = pl_ai.FakeModel()
    df = (
        pl.DataFrame({"topic": ["x", "x"], "_v": ["ok", None]})
        .with_columns(
            pl.struct(
                pl.lit("text").alias("_type"),
                pl.col("_v").cast(pl.Utf8).alias("_value"),
                pl.lit(None).cast(pl.Utf8).alias("_mime"),
                pl.lit("{}").alias("_meta"),
            ).alias("ctx")
        )
        .drop("_v")
        .group_by("topic")
        .agg(pl.col("ctx").ctx.reduce(model=model).alias("out"))
    )
    assert (
        df.select(pl.col("out").ai.status()).to_series().item()
        == pl_ai.RESPONSE_STATUS_INVALID_CONTEXT
    )


def test_ctx_reduce_non_json_meta_field_ok() -> None:
    """Batch cache key still incorporates _meta per atom; reduce succeeds."""
    model = pl_ai.FakeModel(prompt="{value}", tag="meta")
    df = (
        pl.DataFrame({"topic": ["t"], "raw": ["v"]})
        .with_columns(
            pl.struct(
                pl.lit("text").alias("_type"),
                pl.col("raw").alias("_value"),
                pl.lit(None).cast(pl.Utf8).alias("_mime"),
                pl.lit("not-json").alias("_meta"),
            ).alias("ctx")
        )
        .group_by("topic")
        .agg(pl.col("ctx").ctx.reduce(model=model).alias("out"))
    )
    assert df.select(pl.col("out").ai.status()).to_series().item() == pl_ai.RESPONSE_STATUS_OK
    assert len(df.select(pl.col("out").ai.cache_key()).to_series().item()) > 16


def test_ctx_reduce_image_batch_fake_model_error() -> None:
    """Fake provider cannot render multimodal batches."""
    model = pl_ai.FakeModel(prompt="{value}", tag="img")
    tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    df = (
        pl.DataFrame({"g": ["g"], "buf": [tiny_b64]})
        .with_columns(pl_ai.image_context(pl.col("buf"), mime="image/png").alias("ctx"))
        .group_by("g")
        .agg(pl.col("ctx").ctx.reduce(model=model).alias("out"))
    )
    assert (
        df.select(pl.col("out").ai.status()).to_series().item()
        == pl_ai.RESPONSE_STATUS_MODEL_ERROR
    )
    err = df.select(pl.col("out").ai.error()).to_series().item()
    assert err is not None
    assert "multimodal" in err.lower() or "provider" in err.lower()


def test_ctx_reduce_multimodal_false_with_image_model_error() -> None:
    model = pl_ai.FakeModel()
    tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    df = (
        pl.DataFrame({"g": ["g"], "buf": [tiny_b64]})
        .with_columns(pl_ai.image_context(pl.col("buf")).alias("ctx"))
        .group_by("g")
        .agg(pl.col("ctx").ctx.reduce(model=model, multimodal=False).alias("out"))
    )
    assert (
        df.select(pl.col("out").ai.status()).to_series().item()
        == pl_ai.RESPONSE_STATUS_MODEL_ERROR
    )
