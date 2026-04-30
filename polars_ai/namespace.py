"""
polars_ai.namespace
-------------------
Registers the ``.ctx`` namespace on all Polars expressions.

This module is imported as a side-effect by ``polars_ai/__init__.py``.
After ``import polars_ai``, every ``pl.Expr`` gains a ``.ctx`` attribute.

Note on dtype-gating
~~~~~~~~~~~~~~~~~~~~
Polars does not support gating namespaces on dtype at the Python layer —
the same limitation applies to the built-in ``.str``, ``.dt``, ``.list``
namespaces.  Calling ``.ctx`` on a non-AiModelContext column is
syntactically valid but raises a clear error at ``.collect()`` time from
the Rust plugin.

Available methods
~~~~~~~~~~~~~~~~~
    .ctx.preview()         -> pl.Expr[Utf8]    human-readable string
    .ctx.estimate_tokens() -> pl.Expr[UInt32]  rough token estimate
    .ctx.map(model)        -> pl.Expr[Utf8]    run model over each row
"""

from __future__ import annotations

import pathlib
import importlib.machinery

import polars as pl

from .model import AiModel

_IMAGE_TOKEN_ESTIMATE = 1024

# Path to the compiled Rust shared library.
# maturin installs it alongside the Python package as _polars_ai.so / .pyd.
_PACKAGE_DIR = pathlib.Path(__file__).parent


def _lib_path() -> str:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _PACKAGE_DIR / f"_polars_ai{suffix}"
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find the compiled polars-ai plugin. "
        "Run `maturin develop` from the polars-ai directory, then retry."
    )


@pl.api.register_expr_namespace("ctx")
class CtxNamespace:
    """
    Expression namespace for AiModelContext columns.

    Do not instantiate directly — access via ``pl.col("ctx").ctx.*``.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def preview(self) -> pl.Expr:
        """
        Human-readable rendering of the context payload.

        For ``"text"`` contexts returns the value directly.
        For base64 image contexts returns a data URL.

        Returns
        -------
        pl.Expr[Utf8]

        Example
        -------
        >>> df.with_columns(pl.col("ctx").ctx.preview().alias("preview"))
        """
        value = self._expr.struct.field("_value")
        typ = self._expr.struct.field("_type")
        mime = self._expr.struct.field("_mime")

        image_data_url = (
            pl.lit("data:")
            + pl.coalesce([mime, pl.lit("image/jpeg")])
            + pl.lit(";base64,")
            + value
        )
        return (
            pl.when(typ == "text")
            .then(value)
            .when(typ == "image")
            .then(image_data_url)
            .when(typ == "image_url")
            .then(pl.lit("[image_url] ") + value)
            .when(typ == "image_path")
            .then(pl.lit("[image_path] ") + value)
            .otherwise(pl.lit("[unknown] ") + value)
        )

    def estimate_tokens(self) -> pl.Expr:
        """
        Model-agnostic rough token estimate for the context payload.

        For text contexts, uses ``ceil(len(value) / 4)`` which is a
        reasonable approximation for English text with GPT-family
        tokenisers.  For image contexts, returns a fixed conservative
        fallback because exact accounting depends on provider-specific
        image sizing and tiling rules.

        Returns
        -------
        pl.Expr[UInt32]

        Example
        -------
        >>> df.with_columns(pl.col("ctx").ctx.estimate_tokens().alias("tokens"))
        """
        typ = self._expr.struct.field("_type")
        value = self._expr.struct.field("_value")
        text_tokens = value.str.len_chars().truediv(4).ceil()
        image_tokens = pl.lit(_IMAGE_TOKEN_ESTIMATE)

        return (
            pl.when(value.is_null())
            .then(pl.lit(None))
            .when(typ.is_in(["image_url", "image_path", "image"]))
            .then(image_tokens)
            .otherwise(text_tokens)
            .cast(pl.UInt32)
        )

    # ------------------------------------------------------------------
    # Model invocation
    # ------------------------------------------------------------------

    def map(self, model: AiModel) -> pl.Expr:
        """
        Apply *model* to every row of this AiModelContext column.

        Execution is deferred until ``.collect()`` (fully lazy).  At
        collect time the Rust plugin spins up a Tokio runtime, fans out
        all rows concurrently, and throttles dispatch with a built-in
        rate limiter (50 req/s via ``governor``).

        Parameters
        ----------
        model:
            Any :class:`~polars_ai.AiModel` instance.
            Use :class:`~polars_ai.FakeModel` for development / testing.

        Returns
        -------
        pl.Expr[Utf8]

        Example
        -------
        >>> from polars_ai import FakeModel
        >>> model = FakeModel(prompt="Summarise: {value}", tag="v1")
        >>> df.with_columns(pl.col("ctx").ctx.map(model=model).alias("result"))
        """
        from polars.plugins import register_plugin_function  # polars >= 0.41

        return register_plugin_function(
            plugin_path=_lib_path(),
            function_name="ai_map",
            args=[self._expr],
            kwargs={"model_config": model.model_config},
            is_elementwise=True,
        )
