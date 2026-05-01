# Quickstart

Install locally from `polars-ai/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --extras examples
```

Run the notebook sequence:

```bash
marimo edit examples/01_quickstart.py
```

## Sequence

1. `examples/01_quickstart.py`: `FakeModel`, `infer(...)`, and `AiResponse` fields.
1. `examples/02_budgets_cache.py`: request budgets, token budgets, disk cache replay.
1. `examples/03_hydration_real_dataset.py`: partial run and `hydrate(...)`.
1. `examples/04_airbnb_reviews_real_dataset.py`: provider-backed sentiment and weather feature extraction over real Airbnb review text.
1. `examples/05_grouped_and_multimodal.py`: grouped text input and image input options.

## Notes

- `cache_path="cache/"` enables durable disk caching.
- `max_requests`, `max_tokens`, `max_concurrency`, and `rate_limit_per_second` are guardrails applied by the Rust engine.
- Provider examples require API keys; the first three notebooks use `FakeModel` by default.
