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

    result = (
        df.lazy()
        .with_columns(
            pl_ai.text_context(pl.col("review"), format_str="Review: {value}").alias("ctx")
        )
        .with_columns(
            pl.col("ctx").ctx.preview().alias("preview"),
            pl.col("ctx").ctx.estimate_tokens().alias("tokens"),
            pl.col("ctx")
            .ctx.map(model=FakeModel(prompt="Summarise: {value}", tag="demo"))
            .alias("result"),
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

Expression namespace (on any pl.Expr, validated at collect time):
    .ctx.preview()         -> pl.Expr[Utf8]
    .ctx.estimate_tokens() -> pl.Expr[UInt32]
    .ctx.map(model)        -> pl.Expr[Utf8]

Model interface:
    pl_ai.AiModel          abstract base class
    pl_ai.FakeModel        toy implementation, no network calls

Dtype helpers:
    pl_ai.AiModelContext   the pl.Struct constant (for type checks)
    pl_ai.is_context_dtype(dtype) -> bool
"""

# Side-effect: registers the .ctx namespace on pl.Expr
from . import namespace as _namespace  # noqa: F401  (import for side-effect)

# Constructors
from .constructors import context, image_context, text_context

# Model interface
from .model import AiModel, FakeModel

# Dtype constant and helper
from .types import AiModelContext, is_context_dtype

__all__ = [
    # constructors
    "context",
    "text_context",
    "image_context",
    # model
    "AiModel",
    "FakeModel",
    # dtype
    "AiModelContext",
    "is_context_dtype",
]
