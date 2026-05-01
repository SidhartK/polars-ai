"""Provider-backed AI inference as Polars expressions."""

from .api import hydrate, infer
from .model import AnthropicModel, FakeModel, GeminiModel, Model, OpenAIModel
from .types import (
    DEFAULT_CACHE_FOLDER,
    RESPONSE_SCHEMA_VERSION,
    RESPONSE_STATUS_BUDGET_EXHAUSTED,
    RESPONSE_STATUS_CACHE_HIT,
    RESPONSE_STATUS_INVALID_CONTEXT,
    RESPONSE_STATUS_MODEL_ERROR,
    RESPONSE_STATUS_OK,
    RESPONSE_STATUS_PENDING,
    AiResponse,
    assert_response_dtype,
    is_response_dtype,
)

__all__ = [
    "Model",
    "OpenAIModel",
    "AnthropicModel",
    "GeminiModel",
    "FakeModel",
    "infer",
    "hydrate",
    "AiResponse",
    "DEFAULT_CACHE_FOLDER",
    "RESPONSE_SCHEMA_VERSION",
    "RESPONSE_STATUS_OK",
    "RESPONSE_STATUS_CACHE_HIT",
    "RESPONSE_STATUS_PENDING",
    "RESPONSE_STATUS_BUDGET_EXHAUSTED",
    "RESPONSE_STATUS_MODEL_ERROR",
    "RESPONSE_STATUS_INVALID_CONTEXT",
    "is_response_dtype",
    "assert_response_dtype",
]
