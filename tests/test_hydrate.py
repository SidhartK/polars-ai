from __future__ import annotations

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def test_hydrate_fills_incomplete_rows_and_preserves_terminal_rows() -> None:
    model = pl_ai.FakeModel(tag="hyd")
    partial = (
        pl.DataFrame({"msg": ["1", "2", "3"]})
        .lazy()
        .with_columns(pl_ai.infer("msg", model=model, max_requests=1).alias("ai"))
        .collect()
    )

    assert partial.select(pl.col("ai").struct.field("status")).to_series().to_list() == [
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED,
        pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED,
    ]
    first_value = partial.select(pl.col("ai").struct.field("value")).to_series()[0]

    hydrated = (
        partial.lazy()
        .with_columns(
            pl_ai.hydrate(
                response="ai",
                input="msg",
                model=model,
                max_requests=2,
            ).alias("ai")
        )
        .collect()
    )

    assert hydrated.select(pl.col("ai").struct.field("status")).to_series().to_list() == [
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_OK,
    ]
    assert hydrated.select(pl.col("ai").struct.field("value")).to_series()[0] == first_value


def test_hydrate_with_zero_budget_keeps_incomplete_rows_incomplete() -> None:
    model = pl_ai.FakeModel(tag="bud")
    partial = (
        pl.DataFrame({"msg": ["only"]})
        .lazy()
        .with_columns(pl_ai.infer("msg", model=model, max_requests=0).alias("ai"))
        .collect()
    )

    again = (
        partial.lazy()
        .with_columns(
            pl_ai.hydrate(response="ai", input="msg", model=model, max_requests=0).alias("ai")
        )
        .collect()
    )

    assert again.select(pl.col("ai").struct.field("status")).to_series().item() == (
        pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED
    )


def test_chunked_streaming_hydrate_request_budget_is_global() -> None:
    model = pl_ai.FakeModel(tag="global-hydrate")
    chunks = [
        pl.DataFrame({"msg": [f"chunk-{chunk}-row-{row}" for row in range(3)]})
        for chunk in range(3)
    ]
    partial = (
        pl.concat(chunks, rechunk=False)
        .lazy()
        .with_columns(pl_ai.infer("msg", model=model, max_requests=1).alias("ai"))
        .collect(engine="streaming")
    )

    partial_statuses = partial.select(pl.col("ai").struct.field("status")).to_series().to_list()
    assert partial_statuses.count(pl_ai.RESPONSE_STATUS_OK) == 1

    hydrated = (
        partial.lazy()
        .with_columns(
            pl_ai.hydrate(
                response="ai",
                input="msg",
                model=model,
                max_requests=2,
            ).alias("ai")
        )
        .collect(engine="streaming")
    )

    statuses = hydrated.select(pl.col("ai").struct.field("status")).to_series().to_list()
    assert statuses.count(pl_ai.RESPONSE_STATUS_OK) == 3
    assert statuses.count(pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED) == len(statuses) - 3
