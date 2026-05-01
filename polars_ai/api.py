"""Small public inference API."""

from __future__ import annotations

from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

from ._plugin import engine_kwargs, lib_path
from .model import Model

InputType = Literal["text", "image", "image_url", "image_path", "json", "blob", "unknown"]


def _expr(input: pl.Expr | str) -> pl.Expr:
    return pl.col(input) if isinstance(input, str) else input


def _validate_limits(
    *,
    max_requests: int | None,
    max_concurrency: int | None,
    input_type: str,
    text_separator: str,
) -> None:
    if max_requests is not None and max_requests < 0:
        raise ValueError("`max_requests` must be >= 0 or None")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("`max_concurrency` must be >= 1 or None")
    if input_type not in {"text", "image", "image_url", "image_path", "json", "blob", "unknown"}:
        raise ValueError(
            "`input_type` must be one of 'text', 'image', 'image_url', "
            "'image_path', 'json', 'blob', or 'unknown'."
        )
    if not isinstance(text_separator, str):
        raise TypeError("`text_separator` must be a string")


def infer(
    input: pl.Expr | str,
    *,
    model: Model,
    cache_path: str | None = None,
    cache: bool | None = None,
    max_requests: int | None = None,
    max_tokens: int | None = None,
    max_concurrency: int | None = None,
    rate_limit_per_second: int | None = None,
    input_type: InputType = "text",
    mime: str | None = None,
    text_separator: str = "\n\n",
    number_text_items: bool = False,
    multimodal: bool = True,
) -> pl.Expr:
    """Run model inference over a Polars expression and return an ``AiResponse`` struct."""

    _validate_limits(
        max_requests=max_requests,
        max_concurrency=max_concurrency,
        input_type=input_type,
        text_separator=text_separator,
    )
    kwargs = engine_kwargs(
        model=model,
        cache=cache,
        cache_path=cache_path,
        max_requests=max_requests,
        max_tokens=max_tokens,
        max_concurrency=max_concurrency,
        rate_limit_per_second=rate_limit_per_second,
        input_type=input_type,
        mime=mime,
        text_separator=text_separator,
        number_text_items=number_text_items,
        multimodal=multimodal,
    )
    return register_plugin_function(
        plugin_path=lib_path(),
        function_name="ai_infer",
        args=[_expr(input)],
        kwargs=kwargs,
        is_elementwise=True,
    )


def hydrate(
    *,
    response: pl.Expr | str,
    input: pl.Expr | str,
    model: Model,
    cache_path: str | None = None,
    cache: bool | None = None,
    max_requests: int | None = None,
    max_tokens: int | None = None,
    max_concurrency: int | None = None,
    rate_limit_per_second: int | None = None,
    input_type: InputType = "text",
    mime: str | None = None,
    text_separator: str = "\n\n",
    number_text_items: bool = False,
    multimodal: bool = True,
) -> pl.Expr:
    """Complete incomplete ``AiResponse`` rows using the same input/model contract."""

    _validate_limits(
        max_requests=max_requests,
        max_concurrency=max_concurrency,
        input_type=input_type,
        text_separator=text_separator,
    )
    kwargs = engine_kwargs(
        model=model,
        cache=cache,
        cache_path=cache_path,
        max_requests=max_requests,
        max_tokens=max_tokens,
        max_concurrency=max_concurrency,
        rate_limit_per_second=rate_limit_per_second,
        input_type=input_type,
        mime=mime,
        text_separator=text_separator,
        number_text_items=number_text_items,
        multimodal=multimodal,
    )
    return register_plugin_function(
        plugin_path=lib_path(),
        function_name="ai_hydrate",
        args=[_expr(response), _expr(input)],
        kwargs=kwargs,
        is_elementwise=True,
    )
