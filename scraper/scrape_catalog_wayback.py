"""
SHL Individual Test Solutions catalog scraper — Wayback Machine edition.

WHY THIS VERSION EXISTS
------------------------
The live www.shl.com/products/product-catalog/ section (both the paginated
listing page AND individual /view/ detail pages) is currently redirecting
to generic overview pages — confirmed via a real Chrome browser with no
automation involved, so this is not a bot-detection problem. The catalog
section appears to be broken or mid-restructure on the live site as of
this scrape.

The Wayback Machine has indexed this section extensively (SHL's catalog
pages are old, stable, heavily-linked URLs). Strategy:

1. Use the CDX Server API to discover every URL ever archived under
   /products/product-catalog/view/* — this directly gives us the canonical
   list of individual assessment pages, with NO dependency on the listing
   page's pagination at all.
2. For each URL, fetch its most recent successfully-archived (HTTP 200)
   snapshot from web.archive.org, using the `id_` modifier to get the raw
   page content without Wayback's UI toolbar injected.
3. Parse name, test type, duration, languages, and description directly
   off the detail page itself — these are all present there, so we never
   need the listing/index page at all.

KNOWN LIMITATIONS (call these out in your approach doc):
- `remote_testing` / `adaptive_irt` boolean flags were only visible as
  icons in the listing table, which we're bypassing entirely. They're
  left as False/unknown here. If you need them, you'd have to also pull
  an archived snapshot of the listing pages and cross-reference by URL —
  possible but not done here given time constraints; flag it as a known
  gap rather than silently guessing.
- Wayback snapshots may be from different points in time across different
  items (whatever the CDX API's "most recent 200" happens to be per URL),
  so the catalog is not a single consistent point-in-time snapshot. For
  an assignment like this, that's an acceptable and explainable tradeoff
  versus having no data at all.

Usage:
    pip install -r requirements.txt
    python scrape_catalog_wayback.py --out ../data/catalog.json
    # quick sanity check first, on a handful of items:
    python scrape_catalog_wayback.py --out test.json --limit 10
"""
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

CDX_API = "https://web.archive.org/cdx/search/cdx"

TEST_TYPE_NAMES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

_session: Optional[cffi_requests.Session] = None


def _get_session() -> cffi_requests.Session:
    global _session
    if _session is None:
        _session = cffi_requests.Session(impersonate="chrome124")
        _session.headers.update({
            "Accept": "text/html,application/json,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
    return _session


class FetchError(Exception):
    pass


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((FetchError, cffi_requests.RequestsError)),
)
def _get(url: str, params: Optional[dict] = None) -> cffi_requests.Response:
    session = _get_session()
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp
    except cffi_requests.RequestsError as e:
        print(f"  ! request failed for {url}: {type(e).__name__}: {e}", file=sys.stderr)
        raise


def discover_catalog_urls_for_prefix(prefix: str) -> list[dict]:
    """Query the Wayback CDX API for every distinct URL ever archived under
    the given prefix, keeping the most recent HTTP-200 snapshot's timestamp
    for each.

    A trailing "/*" on `url` is auto-interpreted by CDX as matchType=prefix
    (confirmed against the CDX server docs — no separate matchType param
    needed). `limit` is set explicitly rather than relying on the server's
    own default/max, since a silent server-side truncation would show up
    here as "fewer items than expected" with no error to explain why.
    """
    params = {
        "url": prefix,
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,statuscode",
        "limit": "20000",
    }
    resp = _get(CDX_API, params=params)
    rows = resp.json()
    if not rows or len(rows) < 2:
        return []

    header, *data_rows = rows  # first row is the column header
    latest_by_url: dict[str, tuple] = {}
    for ts, original, status in data_rows:
        # Normalize http vs https / trailing slash so we don't end up with
        # near-duplicate items for what's really the same page.
        key = original.split("://", 1)[-1].rstrip("/")
        if key not in latest_by_url or ts > latest_by_url[key][0]:
            latest_by_url[key] = (ts, original)

    return [{"timestamp": ts, "original": original} for ts, original in latest_by_url.values()]


# SHL's catalog detail pages have been observed archived under both of these
# path prefixes (a "/solutions/" insertion at some point in their site's
# history) — query both and merge so we don't silently miss whichever one
# Wayback happened to crawl more of for a given item.
CATALOG_URL_PREFIXES = [
    "shl.com/products/product-catalog/view/*",
    "shl.com/solutions/products/product-catalog/view/*",
]


def discover_catalog_urls() -> list[dict]:
    """Query all known catalog URL prefixes and merge results, deduping by
    the item's slug (the final path segment) so the same assessment found
    under both prefixes only counts once — preferring whichever copy has
    the more recent snapshot timestamp.
    """
    by_slug: dict[str, dict] = {}
    for prefix in CATALOG_URL_PREFIXES:
        print(f"  querying prefix: {prefix}", file=sys.stderr)
        entries = discover_catalog_urls_for_prefix(prefix)
        print(f"    -> {len(entries)} URLs", file=sys.stderr)
        for entry in entries:
            slug = entry["original"].rstrip("/").split("/")[-1]
            if slug not in by_slug or entry["timestamp"] > by_slug[slug]["timestamp"]:
                by_slug[slug] = entry
    return list(by_slug.values())


def fetch_wayback_snapshot(timestamp: str, original_url: str) -> str:
    """Fetch the raw archived HTML for a given snapshot.

    The `id_` suffix on the timestamp tells Wayback to serve the page
    unmodified (no injected toolbar/banner/rewritten links), which keeps
    our BeautifulSoup parsing identical to parsing a normal live page.
    """
    snapshot_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
    resp = _get(snapshot_url)
    return resp.text


DURATION_RE = re.compile(r"(\d+)\s*minute", re.I)
TEST_TYPE_LINE_RE = re.compile(r"test type\s*:?\s*((?:[A-Z](?![a-zA-Z])[\s,]*)+)", re.I)
COMPLETION_RE = re.compile(r"approximate completion time in minutes\s*=?\s*:?\s*(?:max\s+)?(\d+)", re.I)

# Known SHL catalog language labels, longest-first so e.g. "Latin American
# Spanish" matches before the shorter "Spanish" would swallow part of it.
KNOWN_LANGUAGES = sorted([
    "English (USA)", "English International", "English (Global)",
    "English (Australia)", "English (Middle East & North Africa)",
    "English (India)", "English (South Africa)", "English (Canada)",
    "Latin American Spanish", "Spanish", "French (Canada)", "French",
    "German", "Italian", "Dutch", "Portuguese (Brazil)", "Portuguese",
    "Swedish", "Finnish", "Norwegian", "Danish", "Polish", "Czech",
    "Russian", "Turkish", "Greek", "Hungarian", "Romanian",
    "Chinese Simplified", "Chinese Traditional", "Japanese", "Korean",
    "Thai", "Vietnamese", "Indonesian", "Malay", "Hindi", "Arabic",
], key=len, reverse=True)

_LANGUAGE_PATTERN = re.compile(
    "|".join(re.escape(lang) for lang in KNOWN_LANGUAGES)
)

# SHL's site-wide locale-switcher widget (present in header/footer nav on
# EVERY page) surfaces this exact language combination. Treat this specific
# combination as a known false positive and drop it outright if the window
# match is a pure subset of just the widget's languages.
_WIDGET_LANGUAGE_SIGNATURE = {
    "English (Global)", "English (India)", "English (Middle East & North Africa)",
}


def extract_languages(full_text: str) -> list[str]:
    """Find known language names, but ONLY within a window of text near an
    actual 'Language(s)' label on the page — not anywhere in the full body.
    Under-reporting (missing an obscure language) is a much smaller problem
    than over-reporting (confidently attaching a site-nav widget's contents
    to an assessment as if they were its real supported languages)."""
    seen = []
    for label_match in re.finditer(r"languages?\b", full_text, re.I):
        window = full_text[label_match.end(): label_match.end() + 300]
        for match in _LANGUAGE_PATTERN.finditer(window):
            lang = match.group(0)
            if lang not in seen:
                seen.append(lang)
        if seen:
            break

    if set(seen) == _WIDGET_LANGUAGE_SIGNATURE:
        return []

    return seen[:20]


@dataclass
class CatalogItem:
    name: str
    url: str
    test_type: list = field(default_factory=list)
    test_type_labels: list = field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False
    description: str = ""
    job_levels: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    duration_minutes: Optional[int] = None
    duration_raw: str = ""
    source_snapshot_timestamp: str = ""  # transparency: which capture this came from


def parse_detail_page(html: str, fallback_url: str) -> CatalogItem:
    """Extract all fields directly from a detail page — no listing page needed."""
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        name = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        name = soup.title.get_text(strip=True).split("|")[0].strip()
    else:
        name = fallback_url.rstrip("/").split("/")[-1].replace("-", " ").title()

    full_text = soup.get_text(" ", strip=True)

    description = ""
    main = soup.find("main") or soup
    for p in main.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) > 40 and "cookie" not in text.lower():
            description = text
            break

    test_type = []
    m = TEST_TYPE_LINE_RE.search(full_text)
    if m:
        raw_letters = [t.strip() for t in re.split(r"[,\s]+", m.group(1)) if t.strip()]
        seen = set()
        for t in raw_letters:
            if t not in seen:
                seen.add(t)
                test_type.append(t)

    duration_minutes = None
    m = COMPLETION_RE.search(full_text)
    if m:
        duration_minutes = int(m.group(1))
    else:
        m = DURATION_RE.search(full_text)
        if m:
            duration_minutes = int(m.group(1))

    languages = extract_languages(full_text)

    return CatalogItem(
        name=name,
        url=fallback_url,
        test_type=test_type,
        test_type_labels=[TEST_TYPE_NAMES.get(t, t) for t in test_type],
        description=description,
        languages=languages,
        duration_minutes=duration_minutes,
        duration_raw=f"{duration_minutes} minutes" if duration_minutes else "",
    )


def scrape(out_path: str, limit: Optional[int] = None):
    print("Discovering catalog item URLs via Wayback CDX API...", file=sys.stderr)
    discovered = discover_catalog_urls()
    print(f"Found {len(discovered)} distinct archived catalog URLs.", file=sys.stderr)

    if not discovered:
        print("  ! No URLs discovered — check CDX_API reachability / query params.", file=sys.stderr)
        return

    if limit:
        discovered = discovered[:limit]

    items = []
    for i, entry in enumerate(discovered, 1):
        ts, original = entry["timestamp"], entry["original"]
        try:
            html = fetch_wayback_snapshot(ts, original)
            item = parse_detail_page(html, fallback_url=original)
            item.source_snapshot_timestamp = ts
            items.append(asdict(item))
        except Exception as e:
            print(f"  ! failed for {original}: {e}", file=sys.stderr)
        time.sleep(0.3)  # be polite to archive.org too
        if i % 25 == 0:
            print(f"  ...{i}/{len(discovered)}", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Wrote {len(items)} items to {out_path}", file=sys.stderr)

    empty_desc = sum(1 for it in items if not it["description"])
    empty_type = sum(1 for it in items if not it["test_type"])
    print(f"  Sanity check: {empty_desc}/{len(items)} missing description, "
          f"{empty_type}/{len(items)} missing test_type — if these are high, "
          f"the regex patterns above need adjusting against real output.",
          file=sys.stderr)


if __name__ == "__main__":
    # NOTE: deliberately not using argparse here. In a notebook environment
    # (Colab/Jupyter), the kernel launcher injects its own hidden arguments
    # into sys.argv (e.g. "-f /path/to/kernel.json"), which argparse would
    # choke on with an "unrecognized arguments" error the moment you just
    # run the cell — you'd never even get to the scrape. Since this is
    # meant to be pasted into and run from a Colab cell, call scrape()
    # directly instead of trying to parse CLI args.
    #
    # Quick sanity check first (recommended before the full run):
    #   scrape(out_path="test.json", limit=10)
    #
    # Full run:
    #   scrape(out_path="catalog_full.json")
    scrape(out_path="catalog_full.json", limit=10)  # <-- starts with limit=10 as a safety default;
                                                       #     remove limit= once you've confirmed the
                                                       #     output looks right
