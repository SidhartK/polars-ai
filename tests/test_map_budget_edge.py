"""Edge-case integration tests: budgets, hydration, cache keys, lazy slicing (Rust extension required)."""

from __future__ import annotations

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def test_negative_max_requests_raises_on_python_validate() -> None:
    """``ctx.map`` validates budgets when the expression is built."""
    model = pl_ai.FakeModel()
    with pytest.raises(ValueError, match="max_requests"):
        (
            pl.LazyFrame({"x": ["a"]})
            .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
            .with_columns(
                pl.col("ctx").ctx.map(model=model, max_requests=-1).alias("out"),
            )
        )


def test_max_tokens_zero_makes_every_row_budget_exhausted() -> None:
    model = pl_ai.FakeModel()
    stats = (
        pl.LazyFrame({"x": ["a", "bb"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model, max_tokens=0).alias("out"))
        .collect()
        .select(pl.col("out").ai.status())
        .to_series()
        .to_list()
    )
    assert stats == [pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED] * 2


def test_lazy_limit_before_map_truncates_logical_rows() -> None:
    model = pl_ai.FakeModel(prompt="Limited {value}")

    lf = (
        pl.LazyFrame({"id": list(range(1, 21)), "t": ["w"] * 20})
        .with_columns(pl_ai.text_context(pl.col("t")).alias("ctx"))
        .limit(3)
        .with_columns(pl.col("ctx").ctx.map(model=model).alias("out"))
    )
    collected = lf.collect()

    assert collected.height == 3
    assert pl_ai.is_response_dtype(collected["out"].dtype)
    statuses = collected.select(pl.col("out").ai.status()).to_series().to_list()
    assert statuses == [pl_ai.RESPONSE_STATUS_OK] * 3


def test_context_with_null_payload_maps_to_invalid() -> None:
    df = (
        pl.DataFrame({"row": [0]})
        .select(
            pl.struct(
                pl.lit("text").alias("_type"),
                pl.lit(None).cast(pl.Utf8).alias("_value"),
                pl.lit(None).cast(pl.Utf8).alias("_mime"),
                pl.lit("{}").alias("_meta"),
            ).alias("ctx")
        )
    )
    df2 = df.with_columns(
        pl.col("ctx")
        .ctx.map(model=pl_ai.FakeModel(), max_requests=5)
        .alias("out")
    )
    assert df2.select(pl.col("out").ai.status()).to_series()[0] == pl_ai.RESPONSE_STATUS_INVALID_CONTEXT


def test_cache_key_differs_between_models_same_context() -> None:
    lf = (
        pl.LazyFrame({"t": ["shared"]})
        .with_columns(pl_ai.text_context(pl.col("t")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.cache_key(pl_ai.FakeModel(prompt="one", tag="a")).alias("k1"),
            pl.col("ctx").ctx.cache_key(pl_ai.FakeModel(prompt="two", tag="b")).alias("k2"),
        )
    )
    pdf = lf.collect()

    assert pdf["k1"][0] != pdf["k2"][0]


def test_budget_exhausted_rows_keep_non_null_cache_keys() -> None:
    lf = (
        pl.LazyFrame({"x": list("abcde")})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.map(model=pl_ai.FakeModel(), max_requests=2).alias("out"),
        )
    )
    ck = lf.collect().select(pl.col("out").ai.cache_key()).to_series().to_list()
    assert all(isinstance(k, str) and len(k) > 16 for k in ck)


def test_hydrate_with_zero_budget_keeps_budget_exhausted() -> None:
    model = pl_ai.FakeModel(tag="bud")
    first = (
        pl.DataFrame({"x": ["only"]})
        .lazy()
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.map(model=model, max_requests=0).alias("out"),
        )
        .collect()
    )
    assert (
        first.select(pl.col("out").ai.status()).to_series().item()
        == pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED
    )

    again = (
        first.lazy()
        .with_columns(
            pl.col("out").ai.hydrate(ctx=pl.col("ctx"), model=model, max_requests=0).alias(
                "out"
            )
        )
        .collect()
    )
    assert (
        again.select(pl.col("out").ai.status()).to_series().item()
        == pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED
    )


def test_hydrate_keeps_completed_rows_when_new_budget_is_small() -> None:
    """Already ``ok`` responses are terminal; a smaller ``max_requests`` cannot un-fill them."""
    model = pl_ai.FakeModel(prompt="Keep {value}")

    seeded = (
        pl.DataFrame({"x": ["A", "B"]})
        .lazy()
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model, max_requests=2).alias("out"))
        .collect()
    )
    assert all(
        s == pl_ai.RESPONSE_STATUS_OK
        for s in seeded.select(pl.col("out").ai.status()).to_series().to_list()
    )

    after = (
        seeded.lazy()
        .with_columns(
            pl.col("out").ai.hydrate(ctx=pl.col("ctx"), model=model, max_requests=1).alias("out")
        )
        .collect()
    )
    statuses = after.select(pl.col("out").ai.status()).to_series().to_list()
    assert statuses == [pl_ai.RESPONSE_STATUS_OK, pl_ai.RESPONSE_STATUS_OK]


def test_cache_miss_budget_exhausted_without_disk(tmp_path) -> None:
    cache_dir = tmp_path / "empty"
    model = pl_ai.FakeModel()

    statuses = (
        pl.LazyFrame({"x": ["lonely"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.map(
                model=model,
                max_requests=0,
                cache=True,
                cache_path=str(cache_dir),
            ).alias("out"),
        )
        .collect()
        .select(pl.col("out").ai.status())
        .to_series()
        .to_list()
    )

    assert statuses == [pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED]
