"""
polars_ai.types
---------------
AiModelContext dtype constant and helpers.

AiModelContext is NOT a native Polars dtype.  It is a pl.Struct with a
fixed four-field schema that this package treats as a first-class type
by convention — exactly the same way pl.Categorical is a special Utf8.

Users can reference the constant for type checks:

    assert df["ctx"].dtype == pl_ai.AiModelContext
    assert pl_ai.is_context_dtype(df["ctx"].dtype)
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# The canonical schema
# ---------------------------------------------------------------------------
AiModelContext = pl.Struct(
    [
        pl.Field("_type",  pl.Utf8),
        # "text"        — plain string payload
        # "image_url"   — fully-qualified URL; provider fetches it server-side
        # "image_path"  — local file path; read + b64-encoded at request time
        # "image"       — raw bytes stored as Utf8 b64 string (rare)

        pl.Field("_value", pl.Utf8),
        # The payload as a UTF-8 string.
        # For image_path / image_url this is the path / URL.
        # For image bytes this is the base64-encoded string.

        pl.Field("_mime",  pl.Utf8),
        # None for text fields.
        # "image/jpeg", "image/png", etc. for image fields.
        # Required to construct a valid data URL: data:{mime};base64,{b64}

        pl.Field("_meta",  pl.Utf8),
        # JSON string carrying extra metadata, e.g.:
        # {"format_str": "Review: {value}"}
    ]
)

_REQUIRED_FIELDS: frozenset[str] = frozenset({"_type", "_value", "_mime", "_meta"})

# ---------------------------------------------------------------------------
# AiResponse durable result struct — returned by `.ctx.map` / `.ai.hydrate`
# Bump RESPONSE_SCHEMA_VERSION in Rust (`CACHE_SCHEMA_VERSION`) when changing.
# ---------------------------------------------------------------------------
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
    field_names = {f.name for f in dtype.fields}
    return _AI_RESPONSE_REQUIRED.issubset(field_names)


def assert_response_dtype(dtype: pl.PolarsDataType, hint: str = "") -> None:
    """Raise TypeError unless *dtype* is a valid ``AiResponse`` struct."""
    if not is_response_dtype(dtype):
        msg = (
            f"Expected an AiResponse column (got `{dtype}`). "
            f"Use `pl.col(\"...\").ctx.map(...)` or `.ai.hydrate(...)` to create one."
        )
        if hint:
            msg += f"\nHint: {hint}"
        raise TypeError(msg)


def is_context_dtype(dtype: pl.PolarsDataType) -> bool:
    """Return True if *dtype* is a valid AiModelContext Struct."""
    if not isinstance(dtype, pl.Struct):
        return False
    field_names = {f.name for f in dtype.fields}
    return _REQUIRED_FIELDS.issubset(field_names)


def assert_context_dtype(dtype: pl.PolarsDataType, hint: str = "") -> None:
    """Raise a clear TypeError when a non-context dtype reaches a .ctx method."""
    if not is_context_dtype(dtype):
        msg = (
            f"`.ctx` requires an AiModelContext column (got `{dtype}`). "
            f"Use `pl_ai.text_context()`, `pl_ai.image_context()`, or `pl_ai.context()` "
            f"to create one first."
        )
        if hint:
            msg += f"\nHint: {hint}"
        raise TypeError(msg)
