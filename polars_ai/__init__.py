"""
polars_ai
=========
AI model inference as Polars expressions.

Quick start
-----------
    import polars as pl
    import polars_ai as pl_ai
    from polars_ai import FakeModel

    df = pl.DataFrame({"review": ["Great!", "Terrible."]})

    model = FakeModel(prompt="Summarise: {value}", tag="demo")

    result = (
        df.lazy()
        .with_columns(
            pl_ai.text_context(pl.col("review"), format_str="Review: {value}").alias("ctx")
        )
        .with_columns(
            pl.col("ctx").ctx.preview().alias("preview"),
            pl.col("ctx").ctx.estimate_tokens().alias("tokens"),
            pl.col("ctx").ctx.map(model=model).alias("ai_struct"),
            pl.col("ai_struct").ai.value().alias("result"),
        )
        .collect()
    )
    print(result)

Public API
----------
Constructors (return pl.Expr[AiModelContext]):
    pl_ai.context(expr, kind="text", format_str=...)
    pl_ai.context(expr, kind="image", mime=...)
    pl_ai.text_context(expr, format_str=...)
    pl_ai.image_context(expr, mime=...)

Expression namespaces (validated at ``.collect()`` for plugin calls):

    ``.ctx`` on AiModelContext columns:
        .ctx.preview(), .ctx.estimate_tokens(),
        .ctx.map(model=..., budgets...), .ctx.cache_key(model)
    ``.ai`` on AiResponse struct columns produced by ``.ctx.map``:
        .ai.value(), .ai.status(), .ai.telemetry(), .ai.hydrate(ctx=..., model=...)

Model interface:
    pl_ai.AiModel          abstract base class
    pl_ai.FakeModel        toy implementation, no network calls

Dtype helpers:
    pl_ai.AiModelContext   the context pl.Struct convention
    pl_ai.AiResponse       the response pl.Struct convention
"""

# Side-effects: namespaces on pl.Expr (.ctx / .ai)
from . import namespace as _namespace  # noqa: F401
from . import response_namespace as _response_namespace  # noqa: F401

# Constructors
from .constructors import context, image_context, text_context

# Model interface
from .model import AiModel, FakeModel

# Dtype constant and helper
from .types import (
    AiModelContext,
    AiResponse,
    DEFAULT_CACHE_FOLDER,
    RESPONSE_SCHEMA_VERSION,
    RESPONSE_STATUS_BUDGET_EXHAUSTED,
    RESPONSE_STATUS_CACHE_HIT,
    RESPONSE_STATUS_INVALID_CONTEXT,
    RESPONSE_STATUS_MODEL_ERROR,
    RESPONSE_STATUS_OK,
    RESPONSE_STATUS_PENDING,
    assert_response_dtype,
    is_context_dtype,
    is_response_dtype,
)

__all__ = [
    # constructors
    "context",
    "text_context",
    "image_context",
    # model
    "AiModel",
    "FakeModel",
    # constants / dtype
    "AiModelContext",
    "AiResponse",
    "DEFAULT_CACHE_FOLDER",
    "RESPONSE_SCHEMA_VERSION",
    "RESPONSE_STATUS_OK",
    "RESPONSE_STATUS_CACHE_HIT",
    "RESPONSE_STATUS_PENDING",
    "RESPONSE_STATUS_BUDGET_EXHAUSTED",
    "RESPONSE_STATUS_MODEL_ERROR",
    "RESPONSE_STATUS_INVALID_CONTEXT",
    "is_context_dtype",
    "is_response_dtype",
    "assert_response_dtype",
]
