"""Integration tests for `.ctx.map()` with FakeModel (requires compiled extension)."""

from __future__ import annotations

import pytest

pytest.importorskip("polars_ai._polars_ai", reason="Build with `maturin develop` in polars-ai/")

import polars as pl

import polars_ai as pl_ai


def test_ctx_map_fake_model_outputs_per_row_marker() -> None:
    model = pl_ai.FakeModel(prompt="Echo: {value}", tag="test_tag")
    lf = (
        pl.LazyFrame({"msg": ["a", "bbbb"]})
        .with_columns(pl_ai.text_context(pl.col("msg")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model).alias("out"))
    )
    df = lf.collect()

    assert df.height == 2
    outs = df["out"].to_list()
    assert all(isinstance(x, str) for x in outs)
    assert all("[FAKE | tag=test_tag | type=text |" in x for x in outs)
    assert 'prompt="Echo: ' in outs[0]
    assert outs[1].startswith("[FAKE ")


def test_ctx_map_fake_model_preserves_row_order() -> None:
    """Output strings embed per-row payload length matching input order."""
    model = pl_ai.FakeModel(tag="seq")
    rows = ["a", "ab", "abc"]
    df = (
        pl.DataFrame({"n": rows})
        .with_columns(pl_ai.text_context(pl.col("n")).alias("ctx"))
        .with_columns(pl.col("ctx").ctx.map(model=model).alias("out"))
    )
    outs = df["out"].to_list()
    assert len(outs) == 3
    for value, out in zip(rows, outs):
        assert f"input_len={len(value)}]" in out
