"""
``.ai`` expression namespace — helpers for AiResponse struct columns produced by ``.ctx.map`` / hydrate.
"""

from __future__ import annotations

import pathlib
import importlib.machinery
from typing import Any

import polars as pl

from .model import AiModel
from .types import DEFAULT_CACHE_FOLDER

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


def _hydrate_kw(
    *,
    model: AiModel,
    cache: bool | None,
    cache_path: str | None,
    max_requests: int | None,
    max_tokens: int | None,
    max_concurrency: int | None,
    rate_limit_per_second: int | None,
) -> dict[str, Any]:
    resolved = (
        cache_path
        if cache_path is not None
        else (DEFAULT_CACHE_FOLDER if cache else None)
    )
    return {
        "model_config": model.model_config,
        "max_requests": max_requests,
        "max_tokens": max_tokens,
        "max_concurrency": max_concurrency,
        "cache_enabled": cache,
        "cache_path": resolved if cache else None,
        "rate_limit_per_second": rate_limit_per_second,
    }


@pl.api.register_expr_namespace("ai")
class AiNamespace:
    """Operate on AiResponse struct columns."""

    __slots__ = ("_expr",)

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def status(self) -> pl.Expr:
        return self._expr.struct.field("status")

    def value(self) -> pl.Expr:
        return self._expr.struct.field("value")

    def error(self) -> pl.Expr:
        return self._expr.struct.field("error")

    def cache_key(self) -> pl.Expr:
        return self._expr.struct.field("cache_key")

    def attempts(self) -> pl.Expr:
        return self._expr.struct.field("attempts")

    def input_tokens(self) -> pl.Expr:
        return self._expr.struct.field("input_tokens")

    def output_tokens(self) -> pl.Expr:
        return self._expr.struct.field("output_tokens")

    def total_tokens(self) -> pl.Expr:
        return self._expr.struct.field("total_tokens")

    def cost_usd(self) -> pl.Expr:
        return self._expr.struct.field("cost_usd")

    def telemetry(self) -> pl.Expr:
        return pl.struct(
            self.input_tokens(),
            self.output_tokens(),
            self.total_tokens(),
            self.cost_usd(),
        )

    def completed_at(self) -> pl.Expr:
        return self._expr.struct.field("completed_at")

    def is_complete(self) -> pl.Expr:
        import polars_ai.types as t

        stat = self.status()

        incomplete = stat.eq(t.RESPONSE_STATUS_BUDGET_EXHAUSTED)
        return (~incomplete) & stat.is_not_null()

    def hydrate(
        self,
        ctx: pl.Expr,
        model: AiModel,
        *,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        max_concurrency: int | None = None,
        cache: bool = False,
        cache_path: str | None = None,
        rate_limit_per_second: int | None = None,
    ) -> pl.Expr:
        from polars.plugins import register_plugin_function

        if max_requests is not None and max_requests < 0:
            raise ValueError("`max_requests` must be >= 0 or None")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("`max_concurrency` must be >= 1 or None")
        kw = _hydrate_kw(
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
            function_name="ai_hydrate",
            args=[self._expr, ctx],
            kwargs=kw,
            is_elementwise=True,
        )
