import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import re
    import shutil
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl
    import polars_ai as pl_ai

    return Path, go, mo, os, pl, pl_ai


@app.cell
def _(mo):
    mo.md("""
    # 04 - Real Airbnb reviews

    This notebook runs provider-backed LLM inference over real Inside Airbnb
    review text for Cambridge, MA. We extract two features from each review:

    - `sentiment_score`: how positive the stay sounded, from `-1.0` to `1.0`
    - `weather_enjoyment_score`: how much the reviewer enjoyed Cambridge weather,
      from `-1.0` to `1.0`, with `0.0` when weather is not mentioned

    The cells below use the same `polars-ai` primitives as the earlier examples:
    disk caching, request and token budgets, cache-only replay, and hydration.
    """)
    return


@app.cell
def _(Path, mo, os, pl, pl_ai):
    # Change this to "gemini" to use GEMINI_API_KEY or GOOGLE_API_KEY instead.
    LLM_PROVIDER = "openai"

    OPENAI_MODEL = "gpt-4o-mini"
    GEMINI_MODEL = "gemini-1.5-flash"

    SAMPLE_SIZE = 10000
    FIRST_PASS_MAX_REQUESTS = 1000
    HYDRATION_MAX_REQUESTS = 9000
    MAX_TOKENS_PER_PASS = 500_000
    MAX_CONCURRENCY = 50
    RATE_LIMIT_PER_SECOND = 8

    CACHE_DIR = Path(__file__).resolve().parent / "__04_airbnb_cache__"
    # shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = Path(__file__).resolve().parent / "data" / "airbnb_reviews_full.csv"

    FEATURE_PROMPT = """
    You analyze Inside Airbnb guest reviews for Cambridge, Massachusetts.

    Return ONLY valid compact JSON with exactly these keys:
    - sentiment_score: number from -1.0 to 1.0, where -1 is very negative, 0 is neutral, and 1 is very positive.
    - weather_enjoyment_score: number from -1.0 to 1.0 describing how much the reviewer enjoyed Cambridge weather.
      Use 0.0 if the review does not mention weather, season, temperature, rain, snow, sunshine, humidity, or similar outdoor conditions.
    - weather_evidence: short quoted phrase from the review, or an empty string when weather is not mentioned.
    - summary: one short sentence explaining the scores.

    Review:
    {value}

    Response (start your JSON on the line below starting with the character '{'):
    """.strip()

    def make_llm_client(provider):
        provider = provider.lower().strip()
        model_kwargs = {
            "prompt": FEATURE_PROMPT,
            "tag": "airbnb-cambridge-sentiment-weather-v1",
            "temperature": 0.0,
            "max_output_tokens": 180,
        }

        if provider == "openai":
            return pl_ai.OpenAIModel(name=OPENAI_MODEL, **model_kwargs)
        if provider == "gemini":
            return pl_ai.GeminiModel(name=GEMINI_MODEL, **model_kwargs)
        raise ValueError('LLM_PROVIDER must be either "openai" or "gemini"')

    LLM_CLIENT = make_llm_client(LLM_PROVIDER)
    key_status = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
    }

    mo.vstack(
        [
            mo.md("## Configure the model"),
            mo.md(
                f"Using `{LLM_PROVIDER}` via `{LLM_CLIENT.name}`. "
                "The Rust provider layer reads API keys from the environment "
                "when inference runs."
            ),
            pl.DataFrame(
                {
                    "environment_variable": list(key_status.keys()),
                    "is_set": list(key_status.values()),
                }
            ),
        ]
    )
    return (
        CACHE_DIR,
        DATA_PATH,
        FIRST_PASS_MAX_REQUESTS,
        HYDRATION_MAX_REQUESTS,
        LLM_CLIENT,
        MAX_CONCURRENCY,
        MAX_TOKENS_PER_PASS,
        RATE_LIMIT_PER_SECOND,
        SAMPLE_SIZE,
    )


@app.cell
def _(DATA_PATH, SAMPLE_SIZE, mo, pl):
    reviews = (
        pl.read_csv(DATA_PATH)
        .rename({"id": "review_id", "comments": "review_text"})
        .filter(pl.col("review_text").is_not_null())
        .select(
            pl.col("listing_id").cast(pl.Decimal(38, 0)),
            pl.col("review_id").cast(pl.Decimal(38, 0)),
            pl.col("date").cast(pl.Date),
            "reviewer_name",
            "review_text",
        )
        .sample(n=SAMPLE_SIZE, shuffle=True, seed=42)
    )

    mo.vstack(
        [
            mo.md("## Load real review text"),
            mo.md(f"Loaded `{reviews.height}` non-empty reviews from `{DATA_PATH.name}`."),
            reviews,
        ]
    )
    return (reviews,)


@app.cell
def _(reviews):
    reviews.describe()
    return


@app.cell
def _(
    CACHE_DIR,
    FIRST_PASS_MAX_REQUESTS,
    LLM_CLIENT,
    MAX_CONCURRENCY,
    MAX_TOKENS_PER_PASS,
    RATE_LIMIT_PER_SECOND,
    mo,
    pl,
    pl_ai,
    reviews,
):
    first_pass = reviews.with_columns(
        pl_ai.infer(
            "review_text",
            model=LLM_CLIENT,
            cache_path=str(CACHE_DIR),
            max_requests=FIRST_PASS_MAX_REQUESTS,
            max_tokens=MAX_TOKENS_PER_PASS,
            max_concurrency=MAX_CONCURRENCY,
            rate_limit_per_second=RATE_LIMIT_PER_SECOND,
        ).alias("ai")
    )

    first_pass_status = first_pass.group_by(
        pl.col("ai").struct.field("status").alias("status"), maintain_order=True
    ).len()

    mo.vstack(
        [
            mo.md("## First pass: spend a bounded request and token budget"),
            mo.md(
                f"This run makes at most `{FIRST_PASS_MAX_REQUESTS}` new provider "
                f"requests and stops once estimated input tokens exceed `{MAX_TOKENS_PER_PASS}`."
            ),
            first_pass_status,
            first_pass.select(
                "review_id",
                "review_text",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("total_tokens").alias("total_tokens"),
                pl.col("ai").struct.field("cost_usd").alias("cost_usd"),
                pl.col("ai").struct.field("value").alias("raw_features"),
            ).head(8),
        ]
    )
    return (first_pass,)


@app.cell
def _(
    CACHE_DIR,
    HYDRATION_MAX_REQUESTS,
    LLM_CLIENT,
    MAX_CONCURRENCY,
    MAX_TOKENS_PER_PASS,
    RATE_LIMIT_PER_SECOND,
    first_pass,
    mo,
    pl,
    pl_ai,
):
    hydrated = first_pass.with_columns(
        pl_ai.hydrate(
            response="ai",
            input="review_text",
            model=LLM_CLIENT,
            cache_path=str(CACHE_DIR),
            max_requests=HYDRATION_MAX_REQUESTS,
            max_tokens=MAX_TOKENS_PER_PASS,
            max_concurrency=MAX_CONCURRENCY,
            rate_limit_per_second=RATE_LIMIT_PER_SECOND,
        ).alias("ai")
    )

    hydrated_status = hydrated.group_by(
        pl.col("ai").struct.field("status").alias("status"), maintain_order=True
    ).len()

    mo.vstack(
        [
            mo.md("## Hydrate incomplete rows with a fresh budget"),
            mo.md(
                "Re-run this cell to keep filling `budget_exhausted` rows. "
                "Completed rows are preserved and skipped."
            ),
            hydrated_status,
            hydrated.select(
                "review_id",
                pl.col("ai").struct.field("status").alias("status"),
                pl.col("ai").struct.field("total_tokens").alias("total_tokens"),
                pl.col("ai").struct.field("cost_usd").alias("cost_usd"),
                pl.col("ai").struct.field("value").alias("raw_features"),
            ).head(8),
        ]
    )
    return (hydrated,)


@app.cell
def _(
    CACHE_DIR,
    LLM_CLIENT,
    MAX_CONCURRENCY,
    RATE_LIMIT_PER_SECOND,
    hydrated,
    mo,
    pl,
    pl_ai,
    reviews,
):
    cache_replay = reviews.with_columns(
        pl_ai.infer(
            "review_text",
            model=LLM_CLIENT,
            cache_path=str(CACHE_DIR),
            max_requests=0,
            max_concurrency=MAX_CONCURRENCY,
            rate_limit_per_second=RATE_LIMIT_PER_SECOND,
        ).alias("ai")
    )

    replay_status = cache_replay.group_by(
        pl.col("ai").struct.field("status").alias("status"), maintain_order=True
    ).len()

    mo.vstack(
        [
            mo.md("## Replay cached rows without new provider calls"),
            mo.md(
                "`max_requests=0` turns this into a cache-only run. Rows that are "
                "already on disk return `cache_hit`; unfinished rows stay "
                "`budget_exhausted`."
            ),
            replay_status,
            mo.md(
                f"The hydrated table currently has `{hydrated.height}` rows; "
                "use either `hydrated` or `cache_replay` below depending on how "
                "much of the dataset you have completed."
            ),
        ]
    )
    return


@app.cell
def _(hydrated, mo, pl):
    feature_dtype = pl.Struct([
        pl.Field("sentiment_score", pl.Float64), 
        pl.Field("weather_enjoyment_score", pl.Float64), 
        pl.Field("weather_evidence", pl.String), 
        pl.Field("summary", pl.String), 
    ])

    scored_reviews = (
        hydrated.with_columns(
            pl.col("ai")
            .struct.field("value")
            .str.extract(r"(?s)^\s*(?:```(?:json)?\s*)?(\{.*\})\s*(?:```)?\s*$", 1)
            .str.replace(r"\\", "")
            .str.json_decode(feature_dtype)
            .alias("features")
        )
        .unnest("features")
        .with_columns(
            pl.col("ai").struct.field("status").alias("ai_status"),
            pl.col("ai").struct.field("error").alias("ai_error")
        )
    )

    completed_scores = scored_reviews.filter(pl.col("ai_status").is_in(["ok", "cache_hit"]))
    parse_errors = scored_reviews.filter(pl.col("ai_error").is_not_null())

    mo.vstack(
        [
            mo.md("## Scored review table"),
            completed_scores.select(
                "review_id",
                "date",
                "reviewer_name",
                "sentiment_score",
                "weather_enjoyment_score",
                "weather_evidence",
                "summary",
            ).head(20),
            mo.md(
                f"Parsed `{completed_scores.height}` completed rows. "
                f"Rows with JSON parse errors: `{parse_errors.height}`."
            ),
        ]
    )
    return (scored_reviews,)


@app.cell
def _(mo, pl, scored_reviews):
    has_weather_evidence = (
        pl.col("weather_evidence").is_not_null()
        & (pl.col("weather_evidence").str.len_chars() > 0)
    )

    score_summary = scored_reviews.select(
        pl.col("sentiment_score").mean().alias("avg_sentiment_score"),
        pl.col("sentiment_score").std().alias("std_sentiment_score"),
        pl.col("sentiment_score").median().alias("median_sentiment_score"),
        pl.col("weather_enjoyment_score").mean().alias("avg_weather_enjoyment_score"),
        pl.col("weather_enjoyment_score").median().alias(
            "median_weather_enjoyment_score"
        ),
        has_weather_evidence.sum().alias("weather_mentions"),
    )
    status_counts = scored_reviews.group_by("ai_status", maintain_order=True).len()

    weather_mentions = scored_reviews.filter(has_weather_evidence).select(
        "review_id",
        "date",
        "weather_enjoyment_score",
        "weather_evidence",
        "review_text",
    )

    mo.vstack(
        [
            mo.md("### AI response statuses"),
            status_counts,
            mo.md("## Aggregate the extracted features"),
            score_summary,
            mo.md("### Reviews sorted by sentiment (bad to good)"),
            scored_reviews.sort("sentiment_score"),
            mo.md("### Reviews where the model found weather evidence"),
            weather_mentions.sort("weather_enjoyment_score"),
        ]
    )
    return


@app.cell
def _(go, mo, pl, scored_reviews):
    listing_sentiment = (
        scored_reviews
        .filter(
            pl.col("ai_status").is_in(["ok", "cache_hit"])
            & pl.col("sentiment_score").is_not_null()
        )
        .group_by("listing_id")
        .agg(
            pl.len().alias("review_count"),
            pl.col("sentiment_score").mean().alias("avg_sentiment_score"),
            pl.col("date").quantile(0.25).alias("q25_review_date"),
            pl.col("date").quantile(0.75).alias("q75_review_date"),
        )
        .filter(pl.col("review_count") >= 5)
        .with_columns(
            (pl.col("q75_review_date") - pl.col("q25_review_date"))
                .dt.total_days()
                .alias("review_window_days")
        )
        .filter(pl.col("review_window_days") > 0)
        # .with_columns(
        #     pl.col("review_window_days").log().alias("log_review_window_days")
        # )
        .sort("review_count", descending=True)
    )

    listing_sentiment_scatter = go.Figure()
    listing_sentiment_scatter.add_trace(
        go.Scatter(
            x=listing_sentiment["review_window_days"].to_list(),
            y=listing_sentiment["avg_sentiment_score"].to_list(),
            mode="markers",
            # marker=dict(
            #     size=listing_sentiment["review_count"].clip(upper_bound=40).to_list(),
            #     opacity=0.65,
            # ),
            text=listing_sentiment["listing_id"].cast(pl.String).to_list(),
            customdata=list(
                zip(
                    listing_sentiment["review_count"].to_list(),
                    listing_sentiment["review_window_days"].to_list(),
                    listing_sentiment["q25_review_date"].cast(pl.String).to_list(),
                    listing_sentiment["q75_review_date"].cast(pl.String).to_list(),
                )
            ),
            hovertemplate=(
                "Listing=%{text}<br>"
                "Avg sentiment=%{y:.3f}<br>"
                "days q75-q25=%{x:.3f}<br>"
                "Review count=%{customdata[0]}<br>"
                "Window days=%{customdata[1]}<br>"
                "q25=%{customdata[2]}<br>"
                "q75=%{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )
    listing_sentiment_scatter.update_layout(
        title="Listing Sentiment vs. Review Date Spread",
        xaxis_title="q75 review date - q25 review date",
        yaxis_title="Average sentiment score",
        xaxis=dict(type='log'),
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    mo.vstack(
        [
            mo.md("## Listing sentiment by review date spread"),
            mo.ui.plotly(listing_sentiment_scatter),
            listing_sentiment,
        ]
    )
    return


@app.cell
def _(go, mo, pl, scored_reviews):
    monthly_sentiment = (
        scored_reviews
        .filter(
            pl.col("ai_status").is_in(["ok", "cache_hit"])
            & pl.col("sentiment_score").is_not_null()
            & pl.col("weather_enjoyment_score").is_not_null()
        )
        .with_columns(pl.col("date").dt.strftime("%m").alias("review_month"))
        .group_by("review_month", maintain_order=True)
        .agg(
            pl.col("sentiment_score").mean().alias("avg_sentiment_score"),
            pl.col("weather_enjoyment_score").mean().alias("avg_weather_enjoyment_score"),
            pl.len().alias("review_count"),
        )
        .sort("review_month")
    )

    sentiment_trend = go.Figure()
    sentiment_trend.add_trace(
        go.Scatter(
            x=monthly_sentiment["review_month"].to_list(),
            y=monthly_sentiment["avg_sentiment_score"].to_list(),
            mode="lines+markers",
            name="Average sentiment",
            hovertemplate="Month=%{x}<br>Avg sentiment=%{y:.3f}<extra></extra>",
        )
    )
    sentiment_trend.add_trace(
        go.Scatter(
            x=monthly_sentiment["review_month"].to_list(),
            y=monthly_sentiment["avg_weather_enjoyment_score"].to_list(),
            mode="lines+markers",
            name="Average weather sentiment",
            hovertemplate="Month=%{x}<br>Avg weather sentiment=%{y:.3f}<extra></extra>",
        )
    )
    sentiment_trend.update_layout(
        title="Average Airbnb Review Sentiment by Month",
        xaxis_title="Review month",
        yaxis_title="Average score (-1 to 1)",
        yaxis=dict(range=[-1, 1]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=40),
    )

    mo.vstack(
        [
            mo.md("## Monthly sentiment trends"),
            mo.md(
                "This chart aggregates completed review scores by review month, "
                "making it easier to compare overall guest sentiment against "
                "weather-specific sentiment over time."
            ),
            mo.ui.plotly(sentiment_trend),
            monthly_sentiment,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### What to try next

    - Increase `FIRST_PASS_MAX_REQUESTS` or re-run the hydration cell to complete
      more rows.
    - Switch `LLM_PROVIDER` from `"openai"` to `"gemini"` and compare cache keys,
      scores, token usage, and cost estimates.
    - Change the prompt tag when you change the scoring rubric so old cache entries
      do not mix with new semantics.
    """)
    return


if __name__ == "__main__":
    app.run()
