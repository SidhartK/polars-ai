"""Tests for context constructors (no Rust plugin required)."""

from __future__ import annotations

import json

import polars as pl
import pytest

import polars_ai as pl_ai


def test_text_context_default_struct_fields_and_meta() -> None:
    df = pl.DataFrame({"review": ["hello world"]})
    out = df.with_columns(
        pl_ai.text_context(pl.col("review")).alias("ctx")
    )

    unpacked = out.select(
        pl.col("ctx").struct.field("_type"),
        pl.col("ctx").struct.field("_value"),
        pl.col("ctx").struct.field("_mime"),
        pl.col("ctx").struct.field("_meta"),
    )
    rows = unpacked.to_dicts()[0]

    assert rows["_type"] == "text"
    assert rows["_value"] == "hello world"
    assert rows["_mime"] is None

    meta = json.loads(rows["_meta"])
    assert meta == {"format_str": "{value}"}


def test_text_context_custom_format_meta() -> None:
    df = pl.DataFrame({"t": ["x"]})
    out = df.with_columns(
        pl_ai.text_context(pl.col("t"), format_str="Review: {value}").alias("ctx")
    )
    meta = json.loads(out["ctx"][0]["_meta"])
    assert meta == {"format_str": "Review: {value}"}


def test_image_context_default_mime_and_meta() -> None:
    df = pl.DataFrame({"img": ["YmFzZTY0YmFsbA=="]})
    out = df.with_columns(pl_ai.image_context(pl.col("img")).alias("ctx"))
    rows = (
        out.select(
            pl.col("ctx").struct.field("_type"),
            pl.col("ctx").struct.field("_mime"),
        )
        .to_dicts()[0]
    )
    assert rows["_type"] == "image"
    assert rows["_mime"] == "image/jpeg"


def test_image_context_custom_mime() -> None:
    df = pl.DataFrame({"img": ["abc"]})
    out = df.with_columns(
        pl_ai.image_context(pl.col("img"), mime="image/png").alias("ctx")
    )
    mime = out.select(pl.col("ctx").struct.field("_mime")).item()
    assert mime == "image/png"


def test_context_dispatches_image_with_default_mime() -> None:
    df = pl.DataFrame({"buf": ["dGVzdA=="]})
    out = df.with_columns(
        pl_ai.context(pl.col("buf"), kind="image").alias("ctx")
    )
    assert out.select(pl.col("ctx").struct.field("_mime")).item() == "image/jpeg"


def test_context_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind.*text.*image"):
        pl_ai.context(pl.col("x"), kind="video")  # type: ignore[arg-type]
