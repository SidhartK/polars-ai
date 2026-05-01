"""Stable public response schema."""

from __future__ import annotations

import polars as pl

RESPONSE_SCHEMA_VERSION = "2"
DEFAULT_CACHE_FOLDER = "__polars_ai_cache__"

RESPONSE_STATUS_OK = "ok"
RESPONSE_STATUS_CACHE_HIT = "cache_hit"
RESPONSE_STATUS_PENDING = "pending"
RESPONSE_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
RESPONSE_STATUS_MODEL_ERROR = "model_error"
RESPONSE_STATUS_INVALID_CONTEXT = "invalid_context"

_AI_RESPONSE_FIELDS: tuple[tuple[str, pl.PolarsDataType], ...] = (
    ("status", pl.Utf8),
    ("value", pl.Utf8),
    ("cache_key", pl.Utf8),
    ("model_config", pl.Utf8),
    ("error", pl.Utf8),
    ("attempts", pl.UInt32),
    ("input_tokens", pl.UInt64),
    ("output_tokens", pl.UInt64),
    ("total_tokens", pl.UInt64),
    ("cost_usd", pl.Float64),
    ("created_at", pl.Utf8),
    ("completed_at", pl.Utf8),
)

_AI_RESPONSE_REQUIRED: frozenset[str] = frozenset(name for name, _ in _AI_RESPONSE_FIELDS)

AiResponse = pl.Struct([pl.Field(name, dtype) for name, dtype in _AI_RESPONSE_FIELDS])


def is_response_dtype(dtype: pl.PolarsDataType) -> bool:
    """Return True if *dtype* matches ``AiResponse`` (required subset, extra fields OK)."""
    if not isinstance(dtype, pl.Struct):
        return False
    field_names = {field.name for field in dtype.fields}
    return _AI_RESPONSE_REQUIRED.issubset(field_names)


def assert_response_dtype(dtype: pl.PolarsDataType, hint: str = "") -> None:
    """Raise TypeError unless *dtype* is a valid ``AiResponse`` struct."""
    if is_response_dtype(dtype):
        return
    msg = (
        f"Expected an AiResponse column (got `{dtype}`). "
        "Use `pl_ai.infer(...)` or `pl_ai.hydrate(...)` to create one."
    )
    if hint:
        msg += f"\nHint: {hint}"
    raise TypeError(msg)
