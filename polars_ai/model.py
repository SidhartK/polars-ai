"""
polars_ai.model
---------------
Public AiModel interface and built-in FakeModel.

To integrate a real provider, subclass AiModel and implement
``model_config``.  The string it returns is passed verbatim as
``kwargs.model_config`` into the Rust plugin, so pack everything the
Rust side needs into it as JSON.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


class AiModel(ABC):
    """
    Abstract base class for all AI model backends.

    Subclass this to integrate any provider (OpenAI, Anthropic, Gemini,
    Ollama, etc.).

    The only required method is ``model_config``, which must return a
    JSON string that fully describes the invocation.  The Rust plugin
    deserialises this string to reconstruct whatever parameters it needs
    when building and dispatching requests.

    Example
    -------
    >>> class MyModel(AiModel):
    ...     def __init__(self, prompt: str):
    ...         self.prompt = prompt
    ...     @property
    ...     def model_config(self) -> str:
    ...         return json.dumps({"prompt": self.prompt, "endpoint": "..."})
    """

    @property
    @abstractmethod
    def model_config(self) -> str:
        """Return a JSON string fully describing this model invocation."""
        ...


# ---------------------------------------------------------------------------
# FakeModel — deterministic, zero network calls, useful for development
# ---------------------------------------------------------------------------

@dataclass
class FakeModel(AiModel):
    """
    A fake AiModel that returns an f-string incorporating input metadata.

    No network calls are made.  The Rust plugin's ``fake_model_call``
    function uses ``prompt`` and ``tag`` from the serialised config to
    build a response that is semi-unique per row, making it easy to
    inspect the output during development.

    Parameters
    ----------
    prompt:
        Template string.  ``{value}`` is replaced with the row's
        ``_value`` field at response time on the Rust side.
    tag:
        Short label included in every response.  Useful for
        distinguishing outputs from different FakeModel instances
        when chaining calls.

    Example
    -------
    >>> model = FakeModel(prompt="Summarise: {value}", tag="summariser")
    >>> df.with_columns(pl.col("ctx").ctx.map(model=model).alias("result"))
    """

    prompt: str = "Process: {value}"
    tag:    str = "fake"

    @property
    def model_config(self) -> str:
        return json.dumps({"prompt": self.prompt, "tag": self.tag})
