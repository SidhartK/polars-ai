"""Serializable model config objects used by the Rust execution engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

JsonObject = Mapping[str, Any]


def _json_object(value: JsonObject | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"`{field_name}` must be a mapping or None")
    copied = dict(value)
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError(f"`{field_name}` must be JSON serializable") from exc
    return copied


@dataclass
class Model:
    """Deterministic provider + prompt + tag config.

    ``Model`` is intentionally data-only. It is not a plugin interface:
    Rust can only execute providers implemented by the package.
    """

    provider: str
    name: str
    prompt: str = "{value}"
    tag: str = "default"
    options: JsonObject | None = field(default_factory=dict)
    pricing: JsonObject | None = None

    def __post_init__(self) -> None:
        if not self.provider or not self.provider.strip():
            raise ValueError("`provider` must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ValueError("`name` must be a non-empty string")
        if not self.tag or not self.tag.strip():
            raise ValueError("`tag` must be a non-empty string")
        if not isinstance(self.prompt, str):
            raise TypeError("`prompt` must be a string")
        self.options = _json_object(self.options, field_name="options")
        self.pricing = _json_object(self.pricing, field_name="pricing") if self.pricing else None

    @property
    def model_config(self) -> str:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "name": self.name,
            # Kept during the Rust transition; Rust accepts either `name` or `model`.
            "model": self.name,
            "prompt": self.prompt,
            "tag": self.tag,
            "options": self.options,
        }
        if self.pricing is not None:
            payload["pricing"] = self.pricing
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class OpenAIModel(Model):
    """OpenAI model config executed by the Rust provider layer."""

    def __init__(
        self,
        *,
        name: str,
        prompt: str = "{value}",
        tag: str = "openai",
        options: JsonObject | None = None,
        pricing: JsonObject | None = None,
        **provider_options: Any,
    ) -> None:
        merged_options = {**_json_object(options, field_name="options"), **provider_options}
        super().__init__(
            provider="openai",
            name=name,
            prompt=prompt,
            tag=tag,
            options=merged_options,
            pricing=pricing,
        )


class AnthropicModel(Model):
    """Anthropic model config executed by the Rust provider layer."""

    def __init__(
        self,
        *,
        name: str,
        prompt: str = "{value}",
        tag: str = "anthropic",
        options: JsonObject | None = None,
        pricing: JsonObject | None = None,
        **provider_options: Any,
    ) -> None:
        merged_options = {**_json_object(options, field_name="options"), **provider_options}
        super().__init__(
            provider="anthropic",
            name=name,
            prompt=prompt,
            tag=tag,
            options=merged_options,
            pricing=pricing,
        )


class GeminiModel(Model):
    """Gemini model config executed by the Rust provider layer."""

    def __init__(
        self,
        *,
        name: str,
        prompt: str = "{value}",
        tag: str = "gemini",
        options: JsonObject | None = None,
        pricing: JsonObject | None = None,
        **provider_options: Any,
    ) -> None:
        merged_options = {**_json_object(options, field_name="options"), **provider_options}
        super().__init__(
            provider="gemini",
            name=name,
            prompt=prompt,
            tag=tag,
            options=merged_options,
            pricing=pricing,
        )


class FakeModel(Model):
    """Deterministic local model for examples and tests."""

    def __init__(
        self,
        *,
        name: str = "fake",
        prompt: str = "Process: {value}",
        tag: str = "fake",
        options: JsonObject | None = None,
        pricing: JsonObject | None = None,
        **provider_options: Any,
    ) -> None:
        merged_options = {**_json_object(options, field_name="options"), **provider_options}
        super().__init__(
            provider="fake",
            name=name,
            prompt=prompt,
            tag=tag,
            options=merged_options,
            pricing=pricing,
        )

