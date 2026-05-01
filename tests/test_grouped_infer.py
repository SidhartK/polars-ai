from __future__ import annotations

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def test_grouped_list_text_infer_reduces_in_order() -> None:
    model = pl_ai.FakeModel(prompt="Group:\n{value}", tag="group")
    df = (
        pl.DataFrame({"topic": ["A", "A", "B"], "msg": ["x", "y", "z"]})
        .group_by("topic", maintain_order=True)
        .agg(
            pl_ai.infer(
                pl.col("msg").implode(),
                model=model,
                text_separator="\n",
                number_text_items=True,
            ).alias("ai")
        )
        .sort("topic")
    )

    assert df.select(pl.col("ai").struct.field("status")).to_series().to_list() == [
        pl_ai.RESPONSE_STATUS_OK,
        pl_ai.RESPONSE_STATUS_OK,
    ]
    first_value = df.select(pl.col("ai").struct.field("value")).to_series()[0]
    assert "Item 1:\nx\nItem 2:\ny" in first_value


def test_grouped_image_batch_with_fake_model_errors_as_model_error() -> None:
    tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    df = (
        pl.DataFrame({"g": ["g"], "buf": [tiny_b64]})
        .group_by("g")
        .agg(
            pl_ai.infer(
                pl.col("buf").implode(),
                model=pl_ai.FakeModel(tag="img"),
                input_type="image",
                mime="image/png",
            ).alias("ai")
        )
    )

    assert df.select(pl.col("ai").struct.field("status")).to_series().item() == (
        pl_ai.RESPONSE_STATUS_MODEL_ERROR
    )
    error = df.select(pl.col("ai").struct.field("error")).to_series().item()
    assert error is not None
    assert "multimodal" in error.lower() or "provider" in error.lower()
