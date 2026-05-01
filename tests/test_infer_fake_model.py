from __future__ import annotations

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def test_infer_fake_model_outputs_ai_response() -> None:
    model = pl_ai.FakeModel(prompt="Echo: {value}", tag="test_tag")
    df = (
        pl.LazyFrame({"msg": ["a", "bbbb"]})
        .with_columns(pl_ai.infer(pl.col("msg"), model=model).alias("ai"))
        .collect()
    )

    assert pl_ai.is_response_dtype(df["ai"].dtype)
    rows = df.select(
        pl.col("ai").struct.field("status").alias("status"),
        pl.col("ai").struct.field("value").alias("value"),
    ).to_dicts()
    assert [row["status"] for row in rows] == [pl_ai.RESPONSE_STATUS_OK] * 2
    assert "[FAKE | tag=test_tag | type=text |" in rows[0]["value"]
    assert rows[1]["value"].startswith("[FAKE ")


def test_infer_preserves_row_order_and_telemetry() -> None:
    rows = ["a", "ab", "abc"]
    df = (
        pl.DataFrame({"msg": rows})
        .with_columns(pl_ai.infer("msg", model=pl_ai.FakeModel(tag="seq")).alias("ai"))
        .select(
            pl.col("ai").struct.field("value").alias("value"),
            pl.col("ai").struct.field("input_tokens").alias("input_tokens"),
            pl.col("ai").struct.field("output_tokens").alias("output_tokens"),
            pl.col("ai").struct.field("total_tokens").alias("total_tokens"),
            pl.col("ai").struct.field("cost_usd").alias("cost_usd"),
        )
    )

    for source, value in zip(rows, df["value"].to_list()):
        assert value is not None
        assert f"input_len={len(source)}]" in value
    assert df["input_tokens"].to_list() == [1, 1, 1]
    assert all(token > 0 for token in df["output_tokens"].to_list())
    assert df["total_tokens"].to_list() == [
        i + o for i, o in zip(df["input_tokens"].to_list(), df["output_tokens"].to_list())
    ]
    assert df["cost_usd"].to_list() == [0.0, 0.0, 0.0]


def test_budget_and_null_inputs_return_structured_statuses() -> None:
    df = (
        pl.DataFrame({"msg": ["a", "bb", None]}, schema={"msg": pl.Utf8})
        .with_columns(pl_ai.infer("msg", model=pl_ai.FakeModel(), max_requests=1).alias("ai"))
        .select(
            pl.col("ai").struct.field("status").alias("status"),
            pl.col("ai").struct.field("cache_key").alias("cache_key"),
        )
    )

    assert df["status"].to_list() == [
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED,
        pl_ai.RESPONSE_STATUS_INVALID_CONTEXT,
    ]
    assert all(isinstance(key, str) and len(key) > 16 for key in df["cache_key"].to_list())


def test_python_validation_for_budget_knobs() -> None:
    with pytest.raises(ValueError, match="max_requests"):
        pl_ai.infer(pl.col("x"), model=pl_ai.FakeModel(), max_requests=-1)
    with pytest.raises(ValueError, match="max_concurrency"):
        pl_ai.infer(pl.col("x"), model=pl_ai.FakeModel(), max_concurrency=0)
