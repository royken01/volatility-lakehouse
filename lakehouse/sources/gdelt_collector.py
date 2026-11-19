"""GDELT news collector: the headline source of the pre-event sentiment features.

GDELT (gdeltproject.org) is free, key-free and global, records every article it processes with a
15-minute timestamp, and stores the page title from 2019-09-22.  The collector keeps the headlines
that match the topic of the study (``[news] keywords`` in the settings) and writes them as the CSV
that ``vol-lakehouse bronze`` loads (``[sources] headlines``):

    seen_utc, uid, domain, title, tone, score, url

Two back-ends:

  backfill  history, from GDELT 2.0's Global Knowledge Graph raw files (one zipped TSV every 15
            minutes, back to 2015; titles from 2019-09-22).  Multi-process, crash-safe and
            resumable; ``--every N`` downloads only every Nth file.  Only the most relevant stories
            are kept: relevance score >= ``min_score``, near-duplicates removed, at most
            ``--max-per-file`` per 15-minute file.
  doc       recent days, from the DOC 2.0 API (rolling ~1-year window, 250 records per request;
            time windows are bisected until unsaturated).  Use it to top up the most recent
            days.

``seen_utc`` is GDELT's own record time (the end of its 15-minute window); unlike a publisher's
stated time it cannot be edited afterwards, so it is the clock for every no-look-ahead cutoff.

    vol-lakehouse gdelt backfill --start 2019-09-22 --every 8 --max-per-file 3 --dry-run
    vol-lakehouse gdelt backfill --start 2019-09-22 --every 8 --max-per-file 3
    vol-lakehouse gdelt doc --start 2026-11-01
    vol-lakehouse gdelt doc --days 7
    vol-lakehouse gdelt probe
    vol-lakehouse --config configs/example_eurusd_us_macro.toml gdelt backfill --start 2024-01-01

Relevance score (transparent and additive, so it can be reported in a methodology section):
+2 for a keyword hit and +1 for each further distinct keyword (at most +4), +2 per matched
``boost_patterns`` regex, +1 when GDELT tagged a configured theme, +1 for a configured domain,
-3 per ``penalty_patterns`` regex and -4 for auto-generated holdings / ratings notices.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from lakehouse.paths import RAW_DATA_DIR
from lakehouse.settings import Settings, get_settings

log = logging.getLogger("gdelt")
UTC = dt.timezone.utc

HEADLINE_FIELDS = ["seen_utc", "uid", "domain", "title", "tone", "score", "url"]
USER_AGENT = {"User-Agent": "volatility-lakehouse/1.0 (research; https://github.com)"}

GKG_BASE = "http://data.gdeltproject.org/gdeltv2"
GKG_EPOCH = dt.datetime(2015, 2, 18, 23, 0, tzinfo=UTC)
GKG_TITLE_EPOCH = dt.datetime(2019, 9, 22, tzinfo=UTC)   # <PAGE_TITLE> exists from here on
DOC_LOOKBACK_DAYS = 365
# GKG 2.1 column positions
GKG_COL_DATE, GKG_COL_SOURCENAME, GKG_COL_DOCID = 1, 3, 4
GKG_COL_V1THEMES, GKG_COL_V15TONE, GKG_COL_TRANSLATIONINFO, GKG_COL_EXTRASXML = 7, 15, 25, 26
GKG_MIN_COLS = 27
PAGE_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)

SPAM = re.compile(r"\bstake in\b|\b(?:nyse|nasdaq|nysearca|arca|tsx|lse|otcmkts)\s*:|\bshares? (?:bought|sold|acquired)\b|"
                  r"\bposition in\b|\bprice target\b|\bholdings? (?:in|of)\b|\bupgraded\b|\bdowngraded\b|"
                  r"\bshort interest\b", re.I)
TRACKING_PARAMETERS = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|cmpid|ncid|taid|mod|ref|rss)$", re.I)
STOPWORDS = frozenset("the a an and or of in on at to for by with as from after amid despite over into its is are be "
                      "will says say said new not".split())


def require_requests():
    try:
        import requests
    except ImportError as error:
        raise SystemExit("the GDELT collector needs `requests` (pip install requests)") from error
    return requests


# --- small helpers -----------------------------------------------------------------------------
def iso(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).isoformat()


def parse_day(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def parse_compact(stamp: str) -> dt.datetime | None:
    """YYYYMMDDHHMMSS -> aware datetime (None on junk)."""
    stamp = (stamp or "").strip()
    if len(stamp) != 14 or not stamp.isdigit():
        return None
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def normalise_url(url: str) -> str:
    """Drop tracking parameters and fragments so the same article has one uid."""
    url = (url or "").strip()
    if not url.startswith("http"):
        return url
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not TRACKING_PARAMETERS.match(k)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def headline_uid(url: str, title: str) -> str:
    """Stable de-duplication key: sha1 of the normalised URL (or the title)."""
    return hashlib.sha1((normalise_url(url) or title).encode()).hexdigest()[:16]


def story_tokens(text: str) -> frozenset:
    """Word set used for near-duplicate detection (plural-folded, stopwords out)."""
    words = re.findall(r"[a-z0-9$%]+", (text or "").lower())
    return frozenset(w[:-1] if w.endswith("s") and len(w) > 3 else w
                     for w in words if w not in STOPWORDS and len(w) > 1)


def parse_tone(raw: str):
    """First field of GDELT's V1.5TONE (average tone), or '' when malformed."""
    try:
        return float((raw or "").split(",")[0])
    except ValueError:
        return ""


def append_rows(path: Path, rows: Sequence[dict], fields: Sequence[str] = HEADLINE_FIELDS) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def existing_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row.get("uid", "") for row in csv.DictReader(handle)}


def default_output(settings: Settings) -> Path:
    """The headline file bronze reads: ``[sources] headlines`` with any glob characters removed."""
    return RAW_DATA_DIR / re.sub(r"[*?\[\]]", "", settings.headlines_source)


# --- relevance ---------------------------------------------------------------------------------
def _phrase_pattern(phrases: Sequence[str]) -> re.Pattern | None:
    """Whole-word, case-insensitive match of any phrase; a trailing plural 's' is allowed."""
    if not phrases:
        return None
    alternatives = sorted({re.escape(p.lower()).replace(r"\ ", r"\s+") for p in phrases}, key=len, reverse=True)
    return re.compile(r"(?<![\w+])(?:" + "|".join(alternatives) + r")(?:e?s)?(?![\w+])", re.I)


class TopicMatcher:
    """Is a headline about the topic of the study, and how relevant is it?

    ``keywords`` match on their own; an ``ambiguous`` word counts only with one of its context words in
    the same headline ("oil" next to "price", not "olive oil" or "oil painting"); ``mask`` phrases are
    blanked out first so they can never trigger a match.
    """

    def __init__(self, settings: Settings):
        self.keywords = _phrase_pattern(settings.news_keywords)
        self.ambiguous = [(_phrase_pattern([word]), _phrase_pattern(list(context)))
                          for word, context in settings.news_ambiguous]
        self.mask = _phrase_pattern(settings.news_mask)
        self.boost = [re.compile(p, re.I) for p in settings.news_boost_patterns]
        self.penalty = [re.compile(p, re.I) for p in settings.news_penalty_patterns]
        self.themes = tuple(settings.news_themes)
        self.domains = tuple(d.lower() for d in settings.news_domains)
        self.min_score = settings.news_min_score
        if self.keywords is None and not self.ambiguous:
            raise SystemExit("no news keywords configured: set [news] keywords in the config "
                             "(or --news-keywords 'word1,word2')")
        words = list(settings.news_keywords) + [word for word, _ in settings.news_ambiguous]
        self.prefilter = re.compile(
            (r"<PAGE_TITLE>[^<]{0,400}?(?:" + "|".join(re.escape(w.lower()) for w in words) + ")").encode(), re.I)

    def hits(self, text: str) -> set[str]:
        text = self.mask.sub(" ", text) if self.mask else text
        found = {m.group(0).lower() for m in self.keywords.finditer(text)} if self.keywords else set()
        for word, context in self.ambiguous:
            match = word.search(text)
            if match and context is not None and context.search(text[:match.start()] + " " + text[match.end():]):
                found.add(match.group(0).lower())
        return found

    def is_relevant(self, text: str) -> bool:
        return bool(self.hits(text))

    def domain_listed(self, domain: str) -> bool:
        domain = (domain or "").lower().removeprefix("www.")
        return any(domain == d or domain.endswith("." + d) for d in self.domains)

    def score(self, text: str, themes: str = "", domain: str = "") -> int:
        hits = self.hits(text)
        if not hits:
            return 0
        score = 2 + min(len(hits) - 1, 2)
        score += 2 * sum(bool(p.search(text)) for p in self.boost)
        score += 1 if any(t in (themes or "") for t in self.themes) else 0
        score += 1 if self.domain_listed(domain) else 0
        score -= 3 * sum(bool(p.search(text)) for p in self.penalty)
        score -= 4 if SPAM.search(text) else 0
        return score


# --- GKG raw files (backfill) --------------------------------------------------------------------
def gkg_stamps(start: dt.datetime, end: dt.datetime) -> list[str]:
    """Every 15-minute GKG file stamp in [start, end), clamped to the title era."""
    start = max(start.astimezone(UTC), GKG_TITLE_EPOCH)
    end = end.astimezone(UTC)
    current = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)
    stamps = []
    while current < end:
        stamps.append(current.strftime("%Y%m%d%H%M%S"))
        current += dt.timedelta(minutes=15)
    return stamps


def sample_stamps(stamps: Sequence[str], every: int = 1, offset: int = 0) -> list[str]:
    """Every Nth 15-minute file, aligned to the clock (quarter-hour of the day), so re-runs with other
    dates pick the same files and the sample is spread evenly over the day."""
    if every <= 1:
        return list(stamps)
    return [s for s in stamps if (int(s[8:10]) * 4 + int(s[10:12]) // 15) % every == offset % every]


def parse_gkg_blob(blob: bytes, matcher: TopicMatcher, english_only: bool = True) -> list[dict]:
    """One zipped GKG file -> scored candidate headlines, best first (pure: no I/O)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        raw = archive.read(archive.namelist()[0])
    rows = []
    for line in raw.split(b"\n"):
        if not line or not matcher.prefilter.search(line):
            continue
        fields = line.decode("utf-8", errors="replace").rstrip("\r").split("\t")
        if len(fields) <= GKG_COL_V15TONE:
            continue
        fields += [""] * max(0, GKG_MIN_COLS - len(fields))
        url = fields[GKG_COL_DOCID].strip()
        if not url.startswith("http") or (english_only and fields[GKG_COL_TRANSLATIONINFO].strip()):
            continue
        match = PAGE_TITLE.search(fields[GKG_COL_EXTRASXML])
        if not match:
            continue
        title = re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
        domain = fields[GKG_COL_SOURCENAME].strip()
        score = matcher.score(title, fields[GKG_COL_V1THEMES], domain) if title else 0
        seen = parse_compact(fields[GKG_COL_DATE])
        tone = parse_tone(fields[GKG_COL_V15TONE])
        if score < matcher.min_score or seen is None or tone == "":
            continue
        rows.append({"seen_utc": iso(seen), "uid": headline_uid(url, title), "domain": domain, "title": title,
                     "tone": tone, "score": score, "url": url})
    rows.sort(key=lambda row: -row["score"])
    return rows


class StorySelector:
    """Per-file / per-day caps and near-duplicate suppression, applied in file order (causal)."""

    def __init__(self, max_per_file: int, max_per_day: int, jaccard: float, window_hours: float):
        self.max_per_file, self.max_per_day, self.jaccard = max_per_file, max_per_day, jaccard
        self.window = dt.timedelta(hours=window_hours)
        self.by_day: dict[dt.date, list[tuple[dt.datetime, frozenset]]] = {}
        self.day_count: dict[dt.date, int] = {}

    def remember(self, seen: dt.datetime, tokens: frozenset) -> None:
        self.by_day.setdefault(seen.date(), []).append((seen, tokens))
        self.day_count[seen.date()] = self.day_count.get(seen.date(), 0) + 1

    def is_duplicate(self, seen: dt.datetime, tokens: frozenset) -> bool:
        if not tokens or self.jaccard >= 1:
            return False
        for days_back in range(-(self.window.days + 1), 1):
            for earlier, other in self.by_day.get(seen.date() + dt.timedelta(days=days_back), ()):
                if earlier <= seen and seen - earlier <= self.window:
                    union = len(tokens | other)
                    if union and len(tokens & other) / union >= self.jaccard:
                        return True
        return False

    def select(self, candidates: list[dict]) -> list[dict]:
        kept = []
        for candidate in candidates:   # best first
            if len(kept) >= self.max_per_file:
                break
            seen = dt.datetime.fromisoformat(candidate["seen_utc"])
            if self.max_per_day and self.day_count.get(seen.date(), 0) >= self.max_per_day:
                break
            tokens = story_tokens(candidate["title"])
            if self.is_duplicate(seen, tokens):
                continue
            self.remember(seen, tokens)
            kept.append(candidate)
        return kept

    def seed_from_csv(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    self.remember(dt.datetime.fromisoformat(row["seen_utc"]), story_tokens(row.get("title", "")))
                    count += 1
                except (KeyError, ValueError):
                    continue
        return count


_WORKER: dict = {}


def _worker_init(config: dict) -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent flushes and stops the pool on Ctrl+C
    _WORKER.clear()
    _WORKER.update(config)
    _WORKER["matcher"] = TopicMatcher(config["settings"])
    logging.getLogger().setLevel(logging.WARNING)


def _download_and_parse(stamp: str) -> tuple[str, str, int, list[dict]]:
    """Worker: download + parse one GKG file -> (stamp, status, bytes, candidates)."""
    requests = require_requests()
    session = _WORKER.get("session")
    if session is None:
        session = _WORKER["session"] = requests.Session()
        session.headers.update(USER_AGENT)
    url = f"{_WORKER['base_url']}/{stamp}.gkg.csv.zip"
    for attempt in range(1, 5):
        try:
            response = session.get(url, timeout=120)
        except requests.RequestException as error:
            log.warning("GKG %s network error (%d/4): %s", stamp, attempt, error)
            time.sleep(min(60, 2 ** attempt))
            continue
        if response.status_code == 404:
            return stamp, "missing", 0, []          # a genuine gap in GDELT's sequence
        if response.status_code != 200:
            time.sleep(min(60, 2 ** attempt))
            continue
        try:
            candidates = parse_gkg_blob(response.content, _WORKER["matcher"], _WORKER["english_only"])
        except (zipfile.BadZipFile, KeyError, IndexError):
            time.sleep(min(60, 2 ** attempt))
            continue
        return stamp, "ok", len(response.content), candidates[: max(10, 10 * _WORKER["max_per_file"])]
    return stamp, "failed", 0, []


def run_backfill(start: dt.datetime, end: dt.datetime, output: Path, settings: Settings, *, workers: int = 0,
                 every: int = 1, every_offset: int = 0, max_per_file: int = 1, max_per_day: int = 0,
                 dedupe_jaccard: float = 0.6, dedupe_hours: float = 72.0, resume: bool = True,
                 dry_run: bool = False, flush_every: int = 96, base_url: str = GKG_BASE) -> int:
    """Collect relevant headlines from the GKG raw files between ``start`` and ``end`` into ``output``."""
    import multiprocessing

    TopicMatcher(settings)   # fail early on a missing keyword list
    state_file = output.with_name(f".{output.stem}.gkg_done")
    all_stamps = gkg_stamps(start, end)
    stamps = sample_stamps(all_stamps, every, every_offset)
    done = set(state_file.read_text().split()) if resume and state_file.exists() else set()
    pending = [s for s in stamps if s not in done]
    log.info("backfill %s..%s: %d of %d files sampled, %d already done, %d to fetch -> %s",
             start.date(), end.date(), len(stamps), len(all_stamps), len(stamps) - len(pending), len(pending), output)
    if dry_run or not pending:
        return 0
    require_requests()

    selector = StorySelector(max_per_file, max_per_day, dedupe_jaccard, dedupe_hours)
    seeded = selector.seed_from_csv(output)
    if seeded:
        log.info("resume: %d earlier stories loaded for the duplicate check", seeded)
    known = existing_uids(output)
    config = {"settings": settings, "base_url": base_url.rstrip("/"), "max_per_file": max_per_file,
              "english_only": settings.news_english_only}
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 1))
    buffer, buffer_done = [], []
    stats = {"ok": 0, "missing": 0, "failed": 0, "bytes": 0, "kept": 0}
    started = time.monotonic()

    def flush() -> None:
        # data first, state second: a crash in between re-downloads a few files but never marks
        # a file as collected whose rows were not written
        append_rows(output, buffer)
        if buffer_done:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(state_file, "a", encoding="utf-8") as handle:
                handle.write("\n".join(buffer_done) + "\n")
        buffer.clear()
        buffer_done.clear()

    pool = multiprocessing.Pool(workers, initializer=_worker_init, initargs=(config,))
    interrupted = False
    try:
        for index, (stamp, status, size, candidates) in enumerate(pool.imap(_download_and_parse, pending, 2), 1):
            stats[status] += 1
            stats["bytes"] += size
            if status != "failed":
                kept = [row for row in selector.select(candidates) if row["uid"] not in known]
                known.update(row["uid"] for row in kept)
                buffer.extend(kept)
                stats["kept"] += len(kept)
                buffer_done.append(stamp)
            if index % flush_every == 0 or index == len(pending):
                flush()
                rate = index / max(time.monotonic() - started, 1e-9)
                log.info("  %d/%d files (%s) | %.1f GB | %d headlines | %.2f files/s | ETA %.1f h | %d failed",
                         index, len(pending), stamp[:8], stats["bytes"] / 1e9, stats["kept"], rate,
                         (len(pending) - index) / rate / 3600, stats["failed"])
    except KeyboardInterrupt:
        interrupted = True
        log.warning("interrupted - saving what was processed; re-run the same command to resume")
    finally:
        flush()
        pool.terminate()
        pool.join()
    log.info("backfill done: %d ok, %d missing (GDELT gaps), %d failed, %d headlines -> %s",
             stats["ok"], stats["missing"], stats["failed"], stats["kept"], output)
    return 130 if interrupted else (1 if stats["failed"] else 0)


# --- DOC 2.0 API (recent days) -------------------------------------------------------------------
class DocSliceFailed(RuntimeError):
    """A DOC request exhausted its retries (distinct from an empty result)."""


def build_queries(keywords: Sequence[str], group_size: int = 8, sourcelang: str = "english") -> list[str]:
    """OR-groups of the keywords small enough for the DOC API (it rejects long queries)."""
    terms = [f'"{k}"' if " " in k else k for k in keywords]
    return [f"({' OR '.join(terms[i:i + group_size])})" + (f" sourcelang:{sourcelang}" if sourcelang else "")
            for i in range(0, len(terms), group_size)]


class GdeltDocCollector:
    ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
    MAX_RECORDS = 250          # hard API ceiling, no paging

    def __init__(self, settings: Settings, throttle_seconds: float = 6.0, max_retries: int = 3,
                 min_window: dt.timedelta = dt.timedelta(minutes=15)):
        self.matcher = TopicMatcher(settings)
        words = list(settings.news_keywords) + [word for word, _ in settings.news_ambiguous]
        self.queries = build_queries(words, sourcelang="english" if settings.news_english_only else "")
        self.throttle_seconds, self.max_retries, self.min_window = throttle_seconds, max_retries, min_window
        self.session = None
        self.last_request = 0.0
        self.failed_slices: list[dict] = []

    def request(self, query: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
        requests = require_requests()
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update(USER_AGENT)
        parameters = {"query": query, "mode": "artlist", "format": "json", "maxrecords": str(self.MAX_RECORDS),
                      "sort": "datedesc", "startdatetime": start.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
                      "enddatetime": end.astimezone(UTC).strftime("%Y%m%d%H%M%S")}
        reason = ""
        for attempt in range(1, self.max_retries + 1):
            wait = self.throttle_seconds - (time.monotonic() - self.last_request)   # GDELT: <= 1 request / 5 s
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                response = self.session.get(self.ENDPOINT, params=parameters, timeout=60)
            except requests.RequestException as error:
                reason = f"network: {error}"
                time.sleep(min(60, 2 ** attempt * self.throttle_seconds))
                continue
            if response.status_code == 429:
                reason = "HTTP 429 rate limited"
                time.sleep(min(600, 60 * 2 ** (attempt - 1)))
                continue
            if response.status_code in (500, 502, 503, 504):
                reason = f"HTTP {response.status_code}"
                time.sleep(min(120, 2 ** attempt * self.throttle_seconds))
                continue
            if response.status_code != 200:
                raise DocSliceFailed(f"HTTP {response.status_code}: {response.text.strip()[:200]}")
            try:   # malformed queries come back as HTTP 200 with a plain-text error page
                return response.json().get("articles", []) or []
            except (json.JSONDecodeError, ValueError):
                if not response.text.strip():
                    return []
                raise DocSliceFailed(f"non-JSON response: {response.text.strip()[:200]}")
        raise DocSliceFailed(f"gave up after {self.max_retries} attempts ({reason})")

    def fetch_window(self, query: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
        """Bisect a saturated window until each slice is under the 250-record ceiling."""
        articles = self.request(query, start, end)
        if len(articles) < self.MAX_RECORDS or end - start <= self.min_window:
            return articles
        middle = start + (end - start) / 2
        return self.fetch_window(query, start, middle) + self.fetch_window(query, middle, end)

    def to_row(self, article: dict) -> dict | None:
        title = html.unescape((article.get("title") or "").strip())
        url = (article.get("url") or "").strip()
        domain = (article.get("domain") or "").strip()
        seen = parse_compact((article.get("seendate") or "").replace("T", "").replace("Z", ""))
        # the API matches the full text; re-apply the headline rule so DOC rows mean what backfill rows mean
        score = self.matcher.score(title, "", domain) if title and url else 0
        if score < self.matcher.min_score or seen is None:
            return None
        return {"seen_utc": iso(seen), "uid": headline_uid(url, title), "domain": domain, "title": title,
                "tone": "", "score": score, "url": url}

    def collect(self, start: dt.datetime, end: dt.datetime, output: Path) -> int:
        if start < dt.datetime.now(UTC) - dt.timedelta(days=DOC_LOOKBACK_DAYS):
            log.warning("the DOC API only searches ~%d days back; use `backfill` for older dates", DOC_LOOKBACK_DAYS)
        known = existing_uids(output)
        written, cursor = 0, start
        while cursor < end:
            stop = min(cursor + dt.timedelta(days=1), end)
            day_rows = {}
            for number, query in enumerate(self.queries, 1):
                log.info("DOC %s..%s query %d/%d", cursor.date(), stop.date(), number, len(self.queries))
                try:
                    articles = self.fetch_window(query, cursor, stop)
                except DocSliceFailed as error:
                    self.failed_slices.append({"query": query, "start": iso(cursor), "end": iso(stop),
                                               "reason": str(error)})
                    log.error("DOC slice failed (%s..%s): %s - recorded as a gap, not as empty",
                              cursor.date(), stop.date(), error)
                    continue
                for article in articles:
                    row = self.to_row(article)
                    if row and row["uid"] not in known:
                        day_rows[row["uid"]] = row
            written += append_rows(output, list(day_rows.values()))   # saved per day: an abort loses nothing
            known.update(day_rows)
            cursor = stop
        log.info("DOC: %d new headline(s) -> %s (%d failed slices)", written, output, len(self.failed_slices))
        return 1 if self.failed_slices else 0

    def probe(self) -> int:
        """One unretried request over the last hour; prints exactly what GDELT answers."""
        requests = require_requests()
        end = dt.datetime.now(UTC).replace(second=0, microsecond=0)
        parameters = {"query": self.queries[0], "mode": "artlist", "format": "json", "maxrecords": "5",
                      "startdatetime": (end - dt.timedelta(hours=1)).strftime("%Y%m%d%H%M%S"),
                      "enddatetime": end.strftime("%Y%m%d%H%M%S")}
        response = requests.get(self.ENDPOINT, params=parameters, headers=USER_AGENT, timeout=60)
        print(f"HTTP {response.status_code}\n{response.text.strip()[:400] or '<empty>'}")
        return 0 if response.status_code == 200 else 1


# --- command line ---------------------------------------------------------------------------------
def main(arguments: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="vol-lakehouse gdelt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    now = dt.datetime.now(UTC)

    backfill = commands.add_parser("backfill", help="history from the GKG raw files (titles from 2019-09-22)")
    backfill.add_argument("--start", type=parse_day, default=GKG_TITLE_EPOCH)
    backfill.add_argument("--end", type=parse_day, default=now)
    backfill.add_argument("--every", type=int, default=1, help="download only every Nth 15-minute file")
    backfill.add_argument("--every-offset", type=int, default=0)
    backfill.add_argument("--max-per-file", type=int, default=1, help="headlines kept per 15-minute file")
    backfill.add_argument("--max-per-day", type=int, default=0, help="0 = no daily cap")
    backfill.add_argument("--dedupe-jaccard", type=float, default=0.6, help="1 disables near-duplicate removal")
    backfill.add_argument("--workers", type=int, default=0)
    backfill.add_argument("--no-resume", action="store_true")
    backfill.add_argument("--dry-run", action="store_true", help="only count the files")

    doc = commands.add_parser("doc", help="recent days from the DOC 2.0 API")
    doc.add_argument("--start", type=parse_day, default=now - dt.timedelta(days=7))
    doc.add_argument("--end", type=parse_day, default=now)
    doc.add_argument("--days", type=int, help="the last N days (instead of --start)")
    commands.add_parser("probe", help="one raw DOC request (connectivity / rate-limit check)")
    for sub_parser in (backfill, doc):
        sub_parser.add_argument("--out", type=Path, default=default_output(settings),
                                help=f"headline CSV (default: {default_output(settings)})")
    options = parser.parse_args(arguments)

    if options.command == "probe":
        return GdeltDocCollector(settings).probe()
    if options.command == "doc":
        start = options.end - dt.timedelta(days=options.days) if options.days else options.start
        return GdeltDocCollector(settings).collect(start, options.end, options.out)
    return run_backfill(options.start, options.end, options.out, settings, workers=options.workers,
                        every=options.every, every_offset=options.every_offset, max_per_file=options.max_per_file,
                        max_per_day=options.max_per_day, dedupe_jaccard=options.dedupe_jaccard,
                        resume=not options.no_resume, dry_run=options.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
