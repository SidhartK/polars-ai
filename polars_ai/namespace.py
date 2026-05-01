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
    .ctx.map(model, ...)    -> AiResponse struct (budget + telemetry + optional cache)
    .ctx.cache_key(model) -> pl.Expr[Utf8] deterministic key for caching
    .ctx.batch()           -> ContextBatch list for grouped aggregation
"""

from __future__ import annotations

import importlib.machinery
import pathlib
from typing import Any

import polars as pl

from .model import AiModel
from .types import DEFAULT_CACHE_FOLDER

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


def _map_kw(
    *,
    model: AiModel,
    cache: bool,
    cache_path: str | None,
    max_requests: int | None,
    max_tokens: int | None,
    max_concurrency: int | None,
    rate_limit_per_second: int | None,
    multimodal: bool | None = None,
) -> dict[str, Any]:
    resolved_path: str | None
    if cache:
        resolved_path = cache_path if cache_path is not None else DEFAULT_CACHE_FOLDER
    else:
        resolved_path = None

    return {
        "model_config": model.model_config,
        "max_requests": max_requests,
        "max_tokens": max_tokens,
        "max_concurrency": max_concurrency,
        "cache_enabled": cache,
        "cache_path": resolved_path,
        "rate_limit_per_second": rate_limit_per_second,
        "multimodal": multimodal,
    }


def _reduce_text_kw(
    *,
    text_separator: str,
    number_text_items: bool,
) -> dict[str, Any]:
    return {
        "text_separator": text_separator,
        "number_text_items": number_text_items,
    }


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

    def cache_key(self, model: AiModel) -> pl.Expr:
        """Deterministic cache key derived from schema version + model_config + ctx payload."""

        from polars.plugins import register_plugin_function

        return register_plugin_function(
            plugin_path=_lib_path(),
            function_name="ai_cache_key",
            args=[self._expr],
            kwargs={"model_config": model.model_config},
            is_elementwise=True,
        )

    def batch(self) -> pl.Expr:
        """
        Collect row-level contexts into a ContextBatch.

        Use inside ``group_by(...).agg(...)`` to produce one ordered
        ``List[AiModelContext]`` per group.
        """
        return self._expr.implode()

    def map(
        self,
        model: AiModel,
        *,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        max_concurrency: int | None = None,
        cache: bool = False,
        cache_path: str | None = None,
        rate_limit_per_second: int | None = None,
    ) -> pl.Expr:
        """
        Apply *model* to every row (bounded by budgets). Returns an ``AiResponse`` struct.

        Use ``pl.col(\"summary\").ai.value()`` to read textual output once complete.

        Parameters
        ----------
        model :
            Concrete :class:`~polars_ai.AiModel`.
        max_requests :
            Cap on successful provider attempts for this invocation (remaining rows → ``budget_exhausted``).
        max_tokens :
            Rough cumulative token ceiling using the same heuristic as ``estimate_tokens()`` per row.
        max_concurrency :
            Tokio parallelism cap inside the Rust plugin.
        cache :
            If True, reuse ``responses`` stored under ``cache_path``.
        cache_path :
            Relative or absolute folder; defaults to ``__polars_ai_cache__`` next to cwd when ``cache=True``.
        rate_limit_per_second :
            Default 50 tokens/s via governor; ``0`` disables local rate limiting only.

        Returns
        -------
        pl.Expr structured as ``AiResponse`` (status, value, cache_key, ...).

        Example
        -------
        >>> from polars_ai import FakeModel
        >>> model = FakeModel(prompt="Summarise: {value}", tag="v1")
        >>> df.lazy().with_columns(
        ...     pl.col("ctx").ctx.map(model=model, max_requests=2).alias("result")
        ... ).collect()
        """
        from polars.plugins import register_plugin_function

        if max_requests is not None and max_requests < 0:
            raise ValueError("`max_requests` must be >= 0 or None")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("`max_concurrency` must be >= 1 or None")
        kw = _map_kw(
            model=model,
            cache=cache,
            cache_path=cache_path,
            max_requests=max_requests,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
            rate_limit_per_second=rate_limit_per_second,
        )

        return register_plugin_function(
            plugin_path=_lib_path(),
            function_name="ai_map",
            args=[self._expr],
            kwargs=kw,
            is_elementwise=True,
        )

    def reduce(
        self,
        model: AiModel,
        *,
        text_separator: str = "\n\n",
        number_text_items: bool = False,
        multimodal: bool = True,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        max_concurrency: int | None = None,
        cache: bool = False,
        cache_path: str | None = None,
        rate_limit_per_second: int | None = None,
    ) -> pl.Expr:
        """
        Compatibility wrapper for grouped text reduction followed by model invocation.

        Prefer ``.ctx.batch().ctxbatch.reduce_text(...).ctx.map(...)`` for new code.
        """
        reduced = self.batch().ctxbatch.reduce_text(
            text_separator=text_separator,
            number_text_items=number_text_items,
            multimodal=multimodal,
        )
        return reduced.ctx.map(
            model=model,
            max_requests=max_requests,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
            cache=cache,
            cache_path=cache_path,
            rate_limit_per_second=rate_limit_per_second,
        )


@pl.api.register_expr_namespace("ctxbatch")
class CtxBatchNamespace:
    """Expression namespace for ContextBatch ``List[AiModelContext]`` columns."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def len(self) -> pl.Expr:
        """Return the number of context atoms in each batch."""
        return self._expr.list.len()

    def slice(self, offset: int, length: int | None = None) -> pl.Expr:
        """Slice each batch using Polars list semantics."""
        return self._expr.list.slice(offset, length)

    def take(self, k: int) -> pl.Expr:
        """Keep the first *k* atoms from each batch."""
        if k < 0:
            raise ValueError("`k` must be >= 0")
        return self.slice(0, k)

    def reduce_text(
        self,
        *,
        text_separator: str = "\n\n",
        number_text_items: bool = False,
        multimodal: bool = True,
    ) -> pl.Expr:
        """
        Collapse a ContextBatch to a single text ContextAtom.

        Text-like atoms are joined in order. Image atoms are rejected unless
        ``multimodal=False`` is passed, in which case the plugin returns a
        model-style error when the reduced atom is invoked.
        """
        from polars.plugins import register_plugin_function

        if not isinstance(text_separator, str):
            raise TypeError("`text_separator` must be a str")
        kw = _reduce_text_kw(
            text_separator=text_separator,
            number_text_items=number_text_items,
        )
        return register_plugin_function(
            plugin_path=_lib_path(),
            function_name="ai_reduce_text",
            args=[self._expr],
            kwargs=kw,
            is_elementwise=True,
        )

    def cache_key(
        self,
        model: AiModel,
        *,
        multimodal: bool = True,
    ) -> pl.Expr:
        """Deterministic cache key for native batch model invocation."""
        from polars.plugins import register_plugin_function

        return register_plugin_function(
            plugin_path=_lib_path(),
            function_name="ai_batch_cache_key",
            args=[self._expr],
            kwargs={"model_config": model.model_config, "multimodal": multimodal},
            is_elementwise=True,
        )

    def map(
        self,
        model: AiModel,
        *,
        multimodal: bool = True,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        max_concurrency: int | None = None,
        cache: bool = False,
        cache_path: str | None = None,
        rate_limit_per_second: int | None = None,
    ) -> pl.Expr:
        """Invoke a model once per ContextBatch."""
        from polars.plugins import register_plugin_function

        if max_requests is not None and max_requests < 0:
            raise ValueError("`max_requests` must be >= 0 or None")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("`max_concurrency` must be >= 1 or None")
        kw = _map_kw(
            model=model,
            cache=cache,
            cache_path=cache_path,
            max_requests=max_requests,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
            rate_limit_per_second=rate_limit_per_second,
            multimodal=multimodal,
        )

        return register_plugin_function(
            plugin_path=_lib_path(),
            function_name="ai_batch_map",
            args=[self._expr],
            kwargs=kw,
            is_elementwise=True,
        )
