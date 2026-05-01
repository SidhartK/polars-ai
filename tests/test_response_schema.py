from __future__ import annotations

import polars as pl

import polars_ai as pl_ai


def test_ai_response_schema_and_status_constants_are_stable() -> None:
    fields = {field.name: field.dtype for field in pl_ai.AiResponse.fields}

    assert pl_ai.RESPONSE_SCHEMA_VERSION == "2"
    assert fields["status"] == pl.Utf8
    assert fields["value"] == pl.Utf8
    assert fields["cache_key"] == pl.Utf8
    assert fields["model_config"] == pl.Utf8
    assert fields["attempts"] == pl.UInt32
    assert fields["input_tokens"] == pl.UInt64
    assert fields["output_tokens"] == pl.UInt64
    assert fields["total_tokens"] == pl.UInt64
    assert fields["cost_usd"] == pl.Float64

    assert pl_ai.RESPONSE_STATUS_OK == "ok"
    assert pl_ai.RESPONSE_STATUS_CACHE_HIT == "cache_hit"
    assert pl_ai.RESPONSE_STATUS_PENDING == "pending"
    assert pl_ai.RESPONSE_STATUS_BUDGET_EXHAUSTED == "budget_exhausted"
    assert pl_ai.RESPONSE_STATUS_MODEL_ERROR == "model_error"
    assert pl_ai.RESPONSE_STATUS_INVALID_CONTEXT == "invalid_context"


def test_response_dtype_helpers_accept_required_subset_plus_extras() -> None:
    assert pl_ai.is_response_dtype(pl_ai.AiResponse)
    extras = pl.Struct(
        [pl.Field(field.name, field.dtype) for field in pl_ai.AiResponse.fields]
        + [pl.Field("extra", pl.Utf8)]
    )
    assert pl_ai.is_response_dtype(extras)
    assert not pl_ai.is_response_dtype(pl.Utf8)


def test_response_fields_are_plain_polars_struct_access() -> None:
    df = pl.DataFrame({"_": [0]}).select(
        pl.struct(
            pl.lit(pl_ai.RESPONSE_STATUS_OK).alias("status"),
            pl.lit("hello").alias("value"),
            pl.lit("key-1").alias("cache_key"),
            pl.lit("{}").alias("model_config"),
            pl.lit(None).cast(pl.Utf8).alias("error"),
            pl.lit(1, dtype=pl.UInt32).alias("attempts"),
            pl.lit(2, dtype=pl.UInt64).alias("input_tokens"),
            pl.lit(3, dtype=pl.UInt64).alias("output_tokens"),
            pl.lit(5, dtype=pl.UInt64).alias("total_tokens"),
            pl.lit(0.0, dtype=pl.Float64).alias("cost_usd"),
            pl.lit("created").alias("created_at"),
            pl.lit("done").alias("completed_at"),
        ).alias("ai")
    )

    row = df.select(
        pl.col("ai").struct.field("status").alias("status"),
        pl.col("ai").struct.field("value").alias("value"),
        pl.struct(
            pl.col("ai").struct.field("input_tokens"),
            pl.col("ai").struct.field("output_tokens"),
            pl.col("ai").struct.field("total_tokens"),
            pl.col("ai").struct.field("cost_usd"),
        ).alias("telemetry"),
    ).to_dicts()[0]

    assert row == {
        "status": pl_ai.RESPONSE_STATUS_OK,
        "value": "hello",
        "telemetry": {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "cost_usd": 0.0,
        },
    }
