"""Tests for `.ctx.preview()` and `.ctx.estimate_tokens()` (no Rust plugin)."""

from __future__ import annotations

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
