"""Silver GDELT news: cleaning of the raw headline table and incremental sentiment scoring
(the scorer is ``[news] sentiment_model``: CrudeBERT or any Hugging Face sentiment model).

Source: ``bronze.gdelt_headlines``. The scores feed the gold news features
(``medallion/gold/news_features.py``).
"""
import polars as pl

SCORED_SCHEMA = {
    "seen_utc": pl.Datetime("us", "UTC"),
    "uid": pl.Utf8,
    "sentiment_score": pl.Float64,
    "sentiment_ternary": pl.Int32,
    "prob_positive": pl.Float64,
    "prob_negative": pl.Float64,
    "prob_neutral": pl.Float64,
}


def clean_gdelt(raw: pl.DataFrame) -> pl.DataFrame:
    """bronze.gdelt_headlines -> typed, deduplicated headlines.

    Drops rows without a valid timestamp, uid or title, and keeps the
    earliest sighting of each uid.
    """
    parsed = raw.with_columns(
        pl.col("seen_utc").cast(pl.Utf8).str.strip_chars()
        .str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%z", strict=False, time_zone="UTC",
        )
        .alias("seen_utc"),
        pl.col("tone").cast(pl.Float64, strict=False),
        pl.col("score").cast(pl.Int32, strict=False),
        pl.col("uid").cast(pl.Utf8).str.strip_chars(),
    )
    usable = (
        pl.col("seen_utc").is_not_null()
        & pl.col("uid").is_not_null() & (pl.col("uid") != "")
        & pl.col("title").is_not_null()
        & (pl.col("title").str.strip_chars() != "")
    )
    return (
        parsed.filter(usable)
        .select("seen_utc", "uid", "domain", "title", "tone", "score", "url")
        .sort("seen_utc")
        .unique(subset="uid", keep="first", maintain_order=True)
    )


def score_new_headlines(headlines: pl.DataFrame, seen: pl.DataFrame | None,
                        batch_size: int = 32) -> pl.DataFrame:
    """Sentiment scores for the headlines whose uid is not yet in ``seen``.

    Only the new rows are returned, so re-running silver never re-scores old
    headlines.
    """
    if seen is not None and len(seen):
        headlines = headlines.join(seen.select("uid"), on="uid", how="anti")
    if headlines.is_empty():
        return pl.DataFrame(schema=SCORED_SCHEMA)

    from lakehouse.news.sentiment import get_scorer

    print(f"  sentiment: scoring {len(headlines):,} new headline(s)")
    results = get_scorer().score_batch(
        headlines["title"].to_list(), batch_size=batch_size,
    )
    return (
        headlines.select("seen_utc", "uid")
        .with_columns(
            pl.Series(
                "sentiment_score",
                [float(score) for score, _, _ in results],
            ),
            pl.Series(
                "sentiment_ternary",
                [int(ternary) for _, ternary, _ in results],
                dtype=pl.Int32,
            ),
            pl.Series(
                "prob_positive",
                [float(probs["positive"]) for _, _, probs in results],
            ),
            pl.Series(
                "prob_negative",
                [float(probs["negative"]) for _, _, probs in results],
            ),
            pl.Series(
                "prob_neutral",
                [float(probs["neutral"]) for _, _, probs in results],
            ),
        )
    )