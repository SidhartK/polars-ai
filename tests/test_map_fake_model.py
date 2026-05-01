"""Integration tests for `.ctx.map()` with FakeModel (requires compiled extension)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def _pull_values(series: pl.Series) -> list[str | None]:
    return (
        series.to_frame("out").select(pl.col("out").ai.value().alias("v")).get_column("v").to_list()
    )


def test_ctx_map_fake_model_outputs_per_row_marker() -> None:
    model = pl_ai.FakeModel(prompt="Echo: {value}", tag="test_tag")
    lf = (
        pl.LazyFrame({"msg": ["a", "bbbb"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model).alias("out"))
    )
    df = lf.collect()

    assert pl_ai.is_response_dtype(df["out"].dtype)

    vals = _pull_values(df["out"])

    assert df.height == 2
    assert all(isinstance(x, str | type(None)) for x in vals)
    assert vals[0] is not None
    assert "[FAKE | tag=test_tag | type=text |" in vals[0]
    assert vals[1] is not None
    assert vals[1].startswith("[FAKE ")


def test_ctx_map_fake_model_preserves_row_order() -> None:
    model = pl_ai.FakeModel(tag="seq")
    rows = ["a", "ab", "abc"]
    df = (
        pl.DataFrame({"n": rows})
        .with_columns(pl_ai.text_context(pl.col("n")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model).alias("out"))
    )

    vals = _pull_values(df["out"])

    assert len(vals) == 3
    for value, out in zip(rows, vals):
        assert out is not None
        assert f"input_len={len(value)}]" in out


def test_budget_exhausted_after_max_requests() -> None:
    model = pl_ai.FakeModel(prompt=":{value}")
    lf = (
        pl.LazyFrame({"x": ["a", "bb", "ccc"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model, max_requests=2).alias("out"))
    )
    df = lf.collect()

    stat = df.select(pl.col("out").ai.status()).to_series().to_list()
    assert sum(s == pl_ai.RESPONSE_STATUS_OK for s in stat) == 2
    assert stat[2] == pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED


def test_max_requests_zero_budget_exhausted_all_rows() -> None:
    model = pl_ai.FakeModel()
    lf = (
        pl.LazyFrame({"x": ["hi"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model, max_requests=0).alias("out"))
    )
    df = lf.collect()

    stat = df.select(pl.col("out").ai.status()).to_series().item()

    assert stat == pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED


def test_max_tokens_budget() -> None:
    model = pl_ai.FakeModel()
    df = (
        pl.DataFrame({"txt": ["a" * 4, "b" * 4, "c" * 4]})  # each ~= ceil(4/4) token buckets
        .lazy()
        .with_columns(pl_ai.text_context(pl.col("txt")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model, max_tokens=2).alias("out"))
        .collect()
    )
    statuses = df.select(pl.col("out").ai.status()).to_series().to_list()

    ok_count = sum(1 for s in statuses if s == pl_ai.RESPONSE_STATUS_OK)
    be_count = sum(1 for s in statuses if s == pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED)

    assert ok_count == 2
    assert be_count == 1


def test_hydrate_fills_budget_then_idempotent_terminal() -> None:
    model = pl_ai.FakeModel(tag="hyd")
    first = (
        pl.DataFrame({"x": ["1", "2", "3"]})
        .lazy()
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model, max_requests=1).alias("out"))
        .collect()
    )

    statuses1 = first.select(pl.col("out").ai.status()).to_series().to_list()

    assert statuses1[0] == pl_ai.RESPONSE_STATUS_OK
    assert statuses1[1:] == [
        pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED,
        pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED,
    ]

    vals1 = (
        first.select(pl.col("out").ai.value().alias("_v")).get_column("_v").to_list()
    )

    upgraded = (
        first.lazy()
        .with_columns(
            pl.col("out")
            .ai.hydrate(ctx=pl.col("ctx"), model=model, max_requests=2)
            .alias("out")
        )
        .collect()
    )

    statuses2 = upgraded.select(pl.col("out").ai.status()).to_series().to_list()

    # Hydrate ran with ``max_requests=2``: fill two previously exhausted rows.
    assert statuses2 == [
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_OK,
    ]

    vals_rows = upgraded.select(pl.col("out").ai.value().alias("_v")).get_column("_v").to_list()

    assert vals_rows[0] == vals1[0]

    idle = (
        upgraded.lazy()
        .with_columns(
            pl.col("out")
            .ai.hydrate(ctx=pl.col("ctx"), model=model, max_requests=0)
            .alias("out")
        )
        .collect()
    )

    statuses3 = idle.select(pl.col("out").ai.status()).to_series().to_list()

    assert statuses3 == statuses2


def test_disk_cache_returns_cache_hit_on_second_collect_even_with_no_budget(tmp_path: Path) -> None:
    cache_dir = tmp_path / "stash"

    model = pl_ai.FakeModel(prompt="Cache me {value}")

    df1 = (
        pl.LazyFrame({"x": ["one"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.map(
                model=model,
                cache=True,
                cache_path=str(cache_dir),
            ).alias("out")
        )
        .collect()
    )

    df2 = (
        pl.LazyFrame({"x": ["one"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(
            pl.col("ctx").ctx.map(
                model=model,
                max_requests=0,
                cache=True,
                cache_path=str(cache_dir),
            ).alias("out")
        )
        .collect()
    )

    stat1 = df1.select(pl.col("out").ai.status()).to_series().item()
    stat2 = df2.select(pl.col("out").ai.status()).to_series().item()

    assert stat1 == pl_ai.RESPONSE_STATUS_OK

    assert stat2 == pl_ai.RESPONSE_STATUS_CACHE_HIT

    assert df2.select(pl.col("out").ai.value()).to_series().item() == df1.select(
        pl.col("out").ai.value()
    ).to_series().item()


def test_cache_key_stable_across_duplicate_lazy_plans(tmp_path: Path) -> None:
    model = pl_ai.FakeModel(tag="stable")
    q = (
        pl.LazyFrame({"x": ["k"]})
        .with_columns(pl_ai.text_context(pl.col("x")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.cache_key(model=model).alias("k"))
    )
    k1 = q.collect().get_column("k").item()

    assert k1 == q.collect().get_column("k").item()

