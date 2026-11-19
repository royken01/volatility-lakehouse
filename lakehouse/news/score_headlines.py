"""Score a GDELT headline csv with the configured sentiment model, idempotently.

    vol-lakehouse score-headlines --in gdelt_headlines.csv
    vol-lakehouse score-headlines --in gdelt_headlines.csv --sentiment-model ProsusAI/finbert

Relative paths are resolved inside ``lakehouse/data/news``; rows already in the output (keyed on
``uid``) are never re-scored.
"""
import argparse
import csv
import logging
from pathlib import Path

from lakehouse.paths import NEWS_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("score_headlines")

OUTPUT_FIELDS = ["seen_utc", "uid", "title", "tone", "url", "sentiment_score", "sentiment_ternary",
                 "prob_positive", "prob_negative", "prob_neutral"]


def read_headline_rows(path: Path) -> list[dict]:
    """Rows of the headline csv; warns when the file looks re-saved by Excel (rows without a uid
    would be skipped)."""
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    malformed_rows = sum(1 for row in rows if not row.get("uid") or None in row)
    if malformed_rows or any(name and name.endswith(";") for name in (reader.fieldnames or [])):
        log.warning("%s: %d of %d row(s) are malformed - was the file re-saved in Excel? Re-export it from "
                    "gdelt_collector.py instead.", path.name, malformed_rows, len(rows))
    return rows


def already_scored_uids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as file:
        return {row.get("uid", "") for row in csv.DictReader(file)}


def score_file(input_path: Path, output_path: Path, batch_size: int = 32, limit: int = 0) -> int:
    """Append CrudeBERT scores for the not-yet-scored rows of ``input_path`` to ``output_path``;
    returns the number scored."""
    rows = read_headline_rows(input_path)
    scored_uids = already_scored_uids(output_path)
    pending_rows = [row for row in rows if row.get("uid") and row["uid"] not in scored_uids
                    and (row.get("title") or "").strip()]
    if limit:
        pending_rows = pending_rows[:limit]
    log.info("%s: %d row(s), %d already scored, %d to score",
             input_path.name, len(rows), len(scored_uids), len(pending_rows))
    if not pending_rows:
        return 0

    from lakehouse.news.sentiment import get_scorer  # imports torch: only when needed

    scores = get_scorer().score_batch([row["title"] for row in pending_rows], batch_size=batch_size)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with open(output_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, extrasaction="ignore", restval="")
        if needs_header:
            writer.writeheader()
        for row, (score, ternary, probabilities) in zip(pending_rows, scores):
            writer.writerow({
                "seen_utc": row.get("seen_utc", ""), "uid": row["uid"], "title": row["title"],
                "url": row.get("url", ""), "tone": row.get("tone", ""),
                "sentiment_score": round(score, 6), "sentiment_ternary": ternary,
                "prob_positive": round(probabilities["positive"], 6),
                "prob_negative": round(probabilities["negative"], 6),
                "prob_neutral": round(probabilities["neutral"], 6)})
    log.info("wrote %d new score(s) -> %s", len(pending_rows), output_path)
    return len(pending_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="score at most N rows (smoke test)")
    arguments = parser.parse_args()

    input_path = Path(arguments.input_path)
    if not input_path.is_absolute():
        input_path = NEWS_DATA_DIR / input_path
    output_path = Path(arguments.output_path) if arguments.output_path else Path(
        str(input_path).replace(".csv", "_sentiment.csv"))
    if not output_path.is_absolute():
        output_path = NEWS_DATA_DIR / output_path
    score_file(input_path, output_path, arguments.batch_size, arguments.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
