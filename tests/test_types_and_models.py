"""Tests for dtype helpers and model config (no Rust plugin required)."""

from __future__ import annotations

import json

import polars as pl

import polars_ai as pl_ai


def test_is_context_dtype_true_for_constructed_column() -> None:
    df = pl.DataFrame({"t": ["a"]}).with_columns(
        pl_ai.text_context(pl.col("t")).alias("ctx")
    )
    assert pl_ai.is_context_dtype(df["ctx"].dtype)


def test_is_context_dtype_false_for_plain_utf8() -> None:
    assert not pl_ai.is_context_dtype(pl.Utf8)


def test_is_context_dtype_false_for_missing_field() -> None:
    incomplete = pl.Struct(
        [
            pl.Field("_type", pl.Utf8),
            pl.Field("_value", pl.Utf8),
            pl.Field("_mime", pl.Utf8),
            # deliberately omit _meta
        ]
    )
    assert not pl_ai.is_context_dtype(incomplete)


def test_is_context_dtype_accepts_extra_fields() -> None:
    """Forward-compatible: AiModelContext is a convention of required subset."""
    extended = pl.Struct(
        [
            pl.Field("_type", pl.Utf8),
            pl.Field("_value", pl.Utf8),
            pl.Field("_mime", pl.Utf8),
            pl.Field("_meta", pl.Utf8),
            pl.Field("_extra", pl.Utf8),
        ]
    )
    assert pl_ai.is_context_dtype(extended)


def test_fake_model_model_config_roundtrip() -> None:
    model = pl_ai.FakeModel(prompt="Say: {value}", tag="t1")
    cfg = json.loads(model.model_config)
    assert cfg == {"prompt": "Say: {value}", "tag": "t1"}


def test_fake_model_defaults_in_config() -> None:
    model = pl_ai.FakeModel()
    cfg = json.loads(model.model_config)
    assert cfg["prompt"] == "Process: {value}"
    assert cfg["tag"] == "fake"
