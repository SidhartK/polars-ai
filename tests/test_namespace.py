"""Tests for `.ctx.preview()`, `.ctx.estimate_tokens()`, and `.ai` helpers."""

from __future__ import annotations

import pytest
import polars as pl

import polars_ai as pl_ai


def test_preview_text_is_raw_value() -> None:
    df = (
        pl.DataFrame({"review": ["abc"]})
        .with_columns(pl_ai.text_context(pl.col("review")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.preview().alias("preview"))
    )
    assert df["preview"].to_list() == ["abc"]


def test_preview_image_data_url() -> None:
    df = (
        pl.DataFrame({"buf": ["dGVzdA=="]})  # "test" in base64
        .with_columns(
            pl_ai.image_context(pl.col("buf"), mime="image/png").alias("ctx")
        )
        .with_columns(pl.col("ctx").ctx.preview().alias("preview"))
    )
    assert df["preview"].to_list()[0].startswith(
        "data:image/png;base64,dGVzdA=="
    )


def test_preview_unknown_type_fallback() -> None:
    df = (
        pl.DataFrame({"raw": ["x"]})
        .with_columns(
            pl.struct(
                pl.lit("weird").alias("_type"),
                pl.col("raw").alias("_value"),
                pl.lit(None).cast(pl.Utf8).alias("_mime"),
                pl.lit("{}").alias("_meta"),
            ).alias("ctx")
        )
        .with_columns(pl.col("ctx").ctx.preview().alias("preview"))
    )
    assert df["preview"].to_list()[0].startswith("[unknown] ")


def test_estimate_tokens_text_ceil_chars_over_four() -> None:
    # len 10 -> ceil(10/4) = 3
    df = (
        pl.DataFrame({"t": ["x" * 10]})
        .with_columns(pl_ai.text_context(pl.col("t")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.estimate_tokens().alias("tok"))
    )
    assert df["tok"].to_list() == [3]


def test_estimate_tokens_image_fixed_fallback() -> None:
    df = (
        pl.DataFrame({"buf": ["YQ=="]})
        .with_columns(pl_ai.image_context(pl.col("buf")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.estimate_tokens().alias("tok"))
    )
    assert df["tok"].to_list() == [1024]


def test_estimate_tokens_null_value_returns_null_tokens() -> None:
    df = (
        pl.DataFrame({"t": [None]}, schema={"t": pl.Utf8})
        .with_columns(pl_ai.text_context(pl.col("t")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.estimate_tokens().alias("tok"))
    )
    assert df["tok"].to_list() == [None]


def test_ctx_batch_dtype_after_group_agg() -> None:
    df = (
        pl.DataFrame({"g": ["a", "a"], "msg": ["one", "two"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("g", maintain_order=True)
        .agg(pl.col("ctx").ctx.batch().alias("batch"))
    )
    assert pl_ai.is_context_batch_dtype(df["batch"].dtype)
    assert df.select(pl.col("batch").ctxbatch.len()).item() == 2


def test_ctxbatch_take_slice_helpers() -> None:
    df = (
        pl.DataFrame({"g": ["a", "a", "a"], "msg": ["one", "two", "three"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .group_by("g", maintain_order=True)
        .agg(pl.col("ctx").ctx.batch().ctxbatch.take(2).alias("batch"))
    )
    assert df.select(pl.col("batch").ctxbatch.len()).item() == 2


def test_ctx_map_invalid_max_concurrency_raises() -> None:
    """``.ctx.map`` validates ``max_concurrency`` in Python (no Rust needed)."""
    model = pl_ai.FakeModel()
    with pytest.raises(ValueError, match="max_concurrency"):
        (
            pl.LazyFrame({"x": ["a"]})
            .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
            .with_columns(
                pl.col("ctx").ctx.map(model=model, max_concurrency=0).alias("out"),
            )
        )


def test_namespace_methods_lazy_collect() -> None:
    lf = (
        pl.LazyFrame({"s": ["hi"]})
        .with_columns(pl_ai.text_context(pl.col("s")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.preview().alias("preview"),
            pl.col("ctx").ctx.estimate_tokens().alias("tok"),
        )
    )
    df = lf.collect()
    assert df["preview"].to_list() == ["hi"]
    assert df["tok"].to_list() == [1]  # ceil(2/4)=1


def _sample_ai_response_expr() -> pl.Expr:
    return pl.struct(
        pl.lit(pl_ai.RESPONSE_STATUS_OK).alias("status"),
        pl.lit("hello").alias("value"),
        pl.lit("key-1").alias("cache_key"),
        pl.lit('{"tag":"x"}').alias("model_config"),
        pl.lit(None).cast(pl.Utf8).alias("error"),
        pl.lit(1, dtype=pl.UInt32).alias("attempts"),
        pl.lit(12, dtype=pl.UInt64).alias("input_tokens"),
        pl.lit(3, dtype=pl.UInt64).alias("output_tokens"),
        pl.lit(15, dtype=pl.UInt64).alias("total_tokens"),
        pl.lit(0.00042, dtype=pl.Float64).alias("cost_usd"),
        pl.lit("2020-01-01T00:00:00Z").alias("created_at"),
        pl.lit("2020-01-01T00:00:01Z").alias("completed_at"),
    ).alias("resp")


def test_ai_namespace_value_status_on_eager_df() -> None:
    df = pl.DataFrame({"_": [0]}).select(_sample_ai_response_expr())
    df2 = df.select(
        pl.col("resp").ai.status().alias("st"),
        pl.col("resp").ai.value().alias("vl"),
        pl.col("resp").ai.error().alias("er"),
        pl.col("resp").ai.cache_key().alias("ck"),
        pl.col("resp").ai.attempts().alias("at"),
        pl.col("resp").ai.input_tokens().alias("it"),
        pl.col("resp").ai.output_tokens().alias("ot"),
        pl.col("resp").ai.total_tokens().alias("tt"),
        pl.col("resp").ai.cost_usd().alias("cu"),
        pl.col("resp").ai.telemetry().alias("telemetry"),
        pl.col("resp").ai.completed_at().alias("co"),
        pl.col("resp").ai.is_complete().alias("done"),
        pl.col("resp").ai.to_context().alias("ctx"),
    )
    row = df2.to_dicts()[0]
    assert row["st"] == pl_ai.RESPONSE_STATUS_OK
    assert row["vl"] == "hello"
    assert row["er"] is None
    assert row["ck"] == "key-1"
    assert row["at"] == 1
    assert row["it"] == 12
    assert row["ot"] == 3
    assert row["tt"] == 15
    assert row["cu"] == 0.00042
    assert row["telemetry"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "cost_usd": 0.00042,
    }
    assert row["co"] == "2020-01-01T00:00:01Z"
    assert row["done"] is True
    assert row["ctx"]["_type"] == "text"
    assert row["ctx"]["_value"] == "hello"


def test_ai_namespace_lazy_collect() -> None:
    lf = pl.LazyFrame({"_": [0]}).select(_sample_ai_response_expr())
    df = lf.with_columns(pl.col("resp").ai.value().alias("v")).collect()
    assert df["v"].to_list() == ["hello"]


def test_ai_is_complete_budget_exhausted_literal() -> None:
    exh = (
        pl.DataFrame({"_": [1]})
        .select(
            pl.struct(
                pl.lit(pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED).alias("status"),
                pl.lit(None).cast(pl.Utf8).alias("value"),
                pl.lit("").alias("cache_key"),
                pl.lit("{}").alias("model_config"),
                pl.lit(None).cast(pl.Utf8).alias("error"),
                pl.lit(0, dtype=pl.UInt32).alias("attempts"),
                pl.lit(None, dtype=pl.UInt64).alias("input_tokens"),
                pl.lit(None, dtype=pl.UInt64).alias("output_tokens"),
                pl.lit(None, dtype=pl.UInt64).alias("total_tokens"),
                pl.lit(None, dtype=pl.Float64).alias("cost_usd"),
                pl.lit("t").alias("created_at"),
                pl.lit("").alias("completed_at"),
            ).alias("resp")
        )
        .with_columns(pl.col("resp").ai.is_complete().alias("done"))
    )
    assert exh["done"].to_list() == [False]

