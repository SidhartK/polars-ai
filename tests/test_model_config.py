from __future__ import annotations

import json

import pytest

import polars_ai as pl_ai


def test_fake_model_canonical_config_is_deterministic() -> None:
    model = pl_ai.FakeModel(prompt="Say: {value}", tag="t1")

    assert model.model_config == pl_ai.FakeModel(prompt="Say: {value}", tag="t1").model_config
    assert json.loads(model.model_config) == {
        "model": "fake",
        "name": "fake",
        "options": {},
        "prompt": "Say: {value}",
        "provider": "fake",
        "tag": "t1",
    }


def test_provider_models_set_provider_and_options() -> None:
    model = pl_ai.OpenAIModel(
        name="gpt-4o-mini",
        prompt="Classify: {value}",
        tag="sentiment-v1",
        temperature=0.1,
        options={"max_output_tokens": 32},
        pricing={"input_cost_per_1m_tokens": 0.15, "output_cost_per_1m_tokens": 0.60},
    )

    cfg = json.loads(model.model_config)
    assert cfg["provider"] == "openai"
    assert cfg["name"] == "gpt-4o-mini"
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["tag"] == "sentiment-v1"
    assert cfg["options"] == {"max_output_tokens": 32, "temperature": 0.1}
    assert cfg["pricing"]["input_cost_per_1m_tokens"] == 0.15


def test_explicit_supported_provider_configs_exist() -> None:
    assert json.loads(pl_ai.AnthropicModel(name="claude-3-5-haiku-latest").model_config)[
        "provider"
    ] == "anthropic"
    assert json.loads(pl_ai.GeminiModel(name="gemini-1.5-flash").model_config)[
        "provider"
    ] == "gemini"


def test_model_rejects_empty_tag_and_non_json_options() -> None:
    with pytest.raises(ValueError, match="tag"):
        pl_ai.OpenAIModel(name="gpt-4o-mini", tag="")

    with pytest.raises(TypeError, match="options"):
        pl_ai.FakeModel(options={"bad": object()})
