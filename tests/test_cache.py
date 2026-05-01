from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def test_disk_cache_replays_hits_without_request_budget(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    model = pl_ai.FakeModel(prompt="Cache me {value}", tag="cache-v1")

    warm = (
        pl.LazyFrame({"msg": ["one"]})
        .with_columns(pl_ai.infer("msg", model=model, cache_path=str(cache_dir)).alias("ai"))
        .collect()
    )
    replay = (
        pl.LazyFrame({"msg": ["one"]})
        .with_columns(
            pl_ai.infer(
                "msg",
                model=model,
                max_requests=0,
                cache_path=str(cache_dir),
            ).alias("ai")
        )
        .collect()
    )

    assert warm.select(pl.col("ai").struct.field("status")).to_series().item() == (
        pl_ai.RESPONSE_STATUS_OK
    )
    assert replay.select(pl.col("ai").struct.field("status")).to_series().item() == (
        pl_ai.RESPONSE_STATUS_CACHE_HIT
    )
    assert replay.select(pl.col("ai").struct.field("value")).to_series().item() == (
        warm.select(pl.col("ai").struct.field("value")).to_series().item()
    )


def test_cache_keys_change_across_model_tags_and_prompts() -> None:
    df = pl.LazyFrame({"msg": ["shared"]}).with_columns(
        pl_ai.infer("msg", model=pl_ai.FakeModel(prompt="one", tag="a")).alias("a"),
        pl_ai.infer("msg", model=pl_ai.FakeModel(prompt="two", tag="b")).alias("b"),
    ).collect()

    assert df.select(pl.col("a").struct.field("cache_key")).item() != df.select(
        pl.col("b").struct.field("cache_key")
    ).item()
