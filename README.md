# polars_ai

AI model inference as Polars expressions — toy implementation.

For a guided setup, see [`QUICKSTART.md`](QUICKSTART.md).

## Prerequisites

| Tool | Install |
|------|---------|
| Rust toolchain | `curl https://sh.rustup.rs \| sh` |
| Python 3.9+ | system / pyenv / conda |
| maturin | `pip install maturin` |

## Directory layout

```
polars_ai/
├── Cargo.toml              Rust dependencies
├── pyproject.toml          Python build config (maturin)
├── QUICKSTART.md           guided setup and first notebook
├── examples/
│   ├── 01_quickstart.py    marimo quickstart notebook
│   └── static/
│       └── example.jpg     sample image used by the notebook
├── src/
│   └── lib.rs              Rust plugin (Tokio async, governor rate-limit)
└── polars_ai/              Python package
    ├── __init__.py         public API + side-effect namespace registration
    ├── types.py            AiModelContext dtype constant
    ├── constructors.py     text_context(), image_context(), context()
    ├── model.py            AiModel ABC + FakeModel
    └── namespace.py        @register_expr_namespace("ctx")
```

## Setup and run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install maturin
pip install maturin

# 3. Compile the Rust plugin and install it into the venv
maturin develop --extras examples

# 4. Open the quickstart notebook
marimo edit examples/01_quickstart.py
```

`maturin develop` must be re-run every time you edit `src/lib.rs`.

## What the example does

`examples/01_quickstart.py` exercises:

1. Wrapping a text column as `AiModelContext` with `pl_ai.text_context()`
2. Dtype check via `pl_ai.is_context_dtype()`
3. `.ctx.preview()` and `.ctx.estimate_tokens()` (pure Polars, no plugin)
4. `.ctx.map(FakeModel(...))` — lazy, concurrent via Tokio, rate-limited at 50 req/s
5. Chained maps — output of step 1 wrapped as a new context and fed into step 2
6. Base64 image context struct shape
7. `pl_ai.context()` convenience wrapper

## Extending with a real model

Subclass `AiModel` and implement `model_config`:

```python
from polars_ai import AiModel
import json

class OpenAIModel(AiModel):
    def __init__(self, prompt: str, model: str = "gpt-4o-mini"):
        self.prompt = prompt
        self.model  = model

    @property
    def model_config(self) -> str:
        return json.dumps({"prompt": self.prompt, "model": self.model})
```

Then replace `fake_model_call` in `src/lib.rs` with a real `reqwest`
HTTP call to the OpenAI chat completions endpoint, parsing `model` out
of the config JSON to select the model.
