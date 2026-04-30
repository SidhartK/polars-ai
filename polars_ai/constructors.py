"""
polars_ai.constructors
----------------------
Top-level functions that promote plain Polars expressions into
AiModelContext expressions.

Usage
-----
    import polars as pl
    import polars_ai as pl_ai

    pl_ai.text_context(pl.col("review"))
    pl_ai.text_context(pl.col("review"), format_str="Review: {value}")
    pl_ai.image_context(pl.col("image_b64"), mime="image/png")
    pl_ai.context(pl.col("text"), kind="text", format_str="Analyse: {value}")
"""

from __future__ import annotations

import json
from typing import Literal

import polars as pl


def _build_context(
    *,
    typ: str,
    value: pl.Expr,
    mime: str | None,
    meta: dict,
) -> pl.Expr:
    meta_json = json.dumps(meta)
    return pl.struct(
        pl.lit(typ).alias("_type"),
        value.cast(pl.Utf8).alias("_value"),
        pl.lit(mime).cast(pl.Utf8).alias("_mime"),
        pl.lit(meta_json).alias("_meta"),
    )


def context(
    expr: pl.Expr,
    *,
    kind: Literal["text", "image"] = "text",
    format_str: str = "{value}",
    mime: str | None = None,
) -> pl.Expr:
    """
    Construct an AiModelContext from a Polars expression.

    Parameters
    ----------
    expr:
        A ``pl.Utf8``-compatible expression.
        For ``kind="image"`` this must be a base64-encoded image string.
    kind:
        ``"text"`` or ``"image"``.
    format_str:
        How this value is rendered when injected into a prompt.
        ``{value}`` is replaced with the cell content at request time.
        Only used for ``kind="text"``.
    mime:
        MIME type for ``kind="image"`` (e.g. ``"image/png"``). Defaults to
        ``"image/jpeg"`` when not provided.

    Returns
    -------
    pl.Expr
        A ``Struct[_type, _value, _mime, _meta]`` expression.

    Example
    -------
    >>> df.with_columns(
    ...     pl_ai.context(pl.col("review"), kind="text", format_str="Review: {value}").alias("ctx")
    ... )
    """
    if kind == "text":
        return _build_context(
            typ="text",
            value=expr,
            mime=None,
            meta={"format_str": format_str},
        )

    if kind == "image":
        return _build_context(
            typ="image",
            value=expr,
            mime=mime or "image/jpeg",
            meta={"format_str": "[image]"},
        )

    raise ValueError("`kind` must be either 'text' or 'image'.")


def text_context(expr: pl.Expr, *, format_str: str = "{value}") -> pl.Expr:
    """
    Convenience wrapper for ``context(..., kind="text")``.

    Parameters
    ----------
    expr:
        A ``pl.Utf8`` expression.
    format_str:
        How this value is rendered when injected into a prompt.

    Returns
    -------
    pl.Expr
        A ``Struct[_type, _value, _mime, _meta]`` expression.
    """
    return context(expr, kind="text", format_str=format_str)


def image_context(expr: pl.Expr, *, mime: str = "image/jpeg") -> pl.Expr:
    """
    Convenience wrapper for ``context(..., kind="image")``.

    Parameters
    ----------
    expr:
        A ``pl.Utf8`` expression containing a base64-encoded image string.
    mime:
        MIME type for the data URL sent to the provider.

    Returns
    -------
    pl.Expr
        A ``Struct[_type, _value, _mime, _meta]`` expression.
    """
    return context(expr, kind="image", mime=mime)
