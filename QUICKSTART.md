# Quickstart

This guide gets `polars-ai` installed locally and opens the marimo notebook in
`examples/01_quickstart.py`.

## Prerequisites

- Python 3.9+
- Rust toolchain
- `maturin`

Install `maturin` into your environment if you do not already have it:

```bash
pip install maturin
```

## Set Up A Local Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --extras examples
```

`maturin develop` compiles the Rust extension and installs `polars-ai` into the
active virtual environment. Re-run it after editing `src/lib.rs`.

## Open The Notebook

```bash
marimo edit examples/01_quickstart.py
```

The notebook demonstrates:

1. Building text `AiModelContext` columns with `pl_ai.text_context()`
2. Checking context dtypes with `pl_ai.is_context_dtype()`
3. Previewing prompts and estimating tokens
4. Mapping a `FakeModel` over contexts
5. Chaining multiple model maps
6. Creating image contexts from `examples/static/example.jpg`
7. Using the generic `pl_ai.context()` helper

## Optional OpenAI Cells

The quickstart includes OpenAI-backed text and image examples. They are skipped
unless `OPENAI_API_KEY` is set:

```bash
export OPENAI_API_KEY=...
marimo edit examples/01_quickstart.py
```

Without an API key, the notebook still runs the local `FakeModel` examples.
