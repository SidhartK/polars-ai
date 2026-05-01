"""Private helpers for invoking the compiled Polars plugin."""

from __future__ import annotations

import importlib.machinery
import pathlib
import uuid
from typing import Any

from .model import Model
from .types import DEFAULT_CACHE_FOLDER

_PACKAGE_DIR = pathlib.Path(__file__).parent


def lib_path() -> str:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _PACKAGE_DIR / f"_polars_ai{suffix}"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find the compiled polars-ai plugin. "
        "Run `maturin develop` from the polars-ai directory, then retry."
    )


def resolve_cache(*, cache: bool | None, cache_path: str | None) -> tuple[bool, str | None]:
    enabled = (cache_path is not None) if cache is None else cache
    if not enabled:
        return False, None
    return True, cache_path if cache_path is not None else DEFAULT_CACHE_FOLDER


def engine_kwargs(
    *,
    model: Model,
    cache: bool | None,
    cache_path: str | None,
    max_requests: int | None,
    max_tokens: int | None,
    max_concurrency: int | None,
    rate_limit_per_second: int | None,
    input_type: str,
    mime: str | None,
    text_separator: str,
    number_text_items: bool,
    multimodal: bool,
) -> dict[str, Any]:
    cache_enabled, resolved_path = resolve_cache(cache=cache, cache_path=cache_path)
    return {
        "run_id": uuid.uuid4().hex,
        "model_config": model.model_config,
        "max_requests": max_requests,
        "max_tokens": max_tokens,
        "max_concurrency": max_concurrency,
        "cache_enabled": cache_enabled,
        "cache_path": resolved_path,
        "rate_limit_per_second": rate_limit_per_second,
        "input_type": input_type,
        "mime": mime,
        "text_separator": text_separator,
        "number_text_items": number_text_items,
        "multimodal": multimodal,
    }
