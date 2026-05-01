"""Tests for dtype helpers and model config (no Rust plugin required)."""

from __future__ import annotations

import json

import pytest
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


def test_is_response_dtype_matches_canonical_ai_response() -> None:
    assert pl_ai.is_response_dtype(pl_ai.AiResponse)


def test_ai_response_includes_telemetry_fields() -> None:
    fields = {field.name: field.dtype for field in pl_ai.AiResponse.fields}

    assert pl_ai.RESPONSE_SCHEMA_VERSION == "2"
    assert fields["input_tokens"] == pl.UInt64
    assert fields["output_tokens"] == pl.UInt64
    assert fields["total_tokens"] == pl.UInt64
    assert fields["cost_usd"] == pl.Float64


def test_is_response_dtype_false_plain_utf8() -> None:
    assert not pl_ai.is_response_dtype(pl.Utf8)


def test_is_response_dtype_false_without_required_field() -> None:
    missing = pl.Struct(
        [
            pl.Field("status", pl.Utf8),
            pl.Field("value", pl.Utf8),
            # omit cache_key ...
        ]
    )
    assert not pl_ai.is_response_dtype(missing)


def test_is_response_dtype_accepts_extra_fields() -> None:
    extras = pl.Struct(
        [pl.Field(f.name, f.dtype) for f in pl_ai.AiResponse.fields]
        + [pl.Field("extra", pl.Utf8)]
    )
    assert pl_ai.is_response_dtype(extras)


def test_assert_response_dtype_raises() -> None:
    with pytest.raises(TypeError):
        pl_ai.assert_response_dtype(pl.Utf8)


def test_status_constants_stable() -> None:
    """Keep literal strings exported for callers and parquet cache interoperability."""
    assert pl_ai.RESPONSE_STATUS_OK == "ok"
    assert pl_ai.RESPONSE_STATUS_CACHE_HIT == "cache_hit"
    assert pl_ai.RESPONSE_STATUS_PENDING == "pending"
    assert pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED == "budget_exhausted"
    assert pl_ai.RESPONSE_STATUS_MODEL_ERROR == "model_error"
    assert pl_ai.RESPONSE_STATUS_INVALID_CONTEXT == "invalid_context"


def test_fake_model_model_config_roundtrip() -> None:
    model = pl_ai.FakeModel(prompt="Say: {value}", tag="t1")
    cfg = json.loads(model.model_config)
    assert cfg == {"prompt": "Say: {value}", "tag": "t1"}


def test_fake_model_defaults_in_config() -> None:
    model = pl_ai.FakeModel()
    cfg = json.loads(model.model_config)
    assert cfg["prompt"] == "Process: {value}"
    assert cfg["tag"] == "fake"

