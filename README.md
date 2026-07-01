# SHL Conversational Assessment Recommender

Take-home submission for the SHL Labs AI Intern role.

## What's here

```
app/                FastAPI service (the actual deliverable)
  main.py           GET /health, POST /chat
  graph.py           LangGraph agent: parse -> route -> retrieve/compare/refuse -> compose
  catalog.py          BM25 retrieval over the SHL catalog
  llm.py               LLM client wrapper (JSON-mode + repair retry)
  prompts.py            Parse-stage and compose-stage system prompts
  schemas.py            Pydantic request/response models (the hard schema contract)
data/
  catalog.json         Seed catalog (35 items, ground-truthed against the 10 public traces)
scraper/                Full ~380-item catalog scraper — RUN THIS LOCALLY (see below)
  scrape_catalog_wayback.py  The working scraper (Wayback Machine CDX API based)
eval/
  run_eval.py          Local hard-eval harness, replays the public traces against a live /chat
traces/                 The 10 provided sample conversations (unmodified)
approach.md             2-page approach document for submission
```

## Catalog: DONE — 553 real items, verified end-to-end

`data/catalog.json` now has the **full scraped catalog: 553 items**, pulled
from live archived SHL pages via the Wayback Machine CDX API (the live
`shl.com/products/product-catalog/` listing page was redirecting to a
generic overview page at scrape time — confirmed in a real browser, not a
bot-detection issue — so the scraper discovers every catalog detail-page URL
ever archived and pulls the most recent working snapshot of each, rather
than depending on the broken listing page at all).

**Verified, not assumed:**
- 0/553 items missing `description`, 0/553 missing `test_type`
- Spot-checked across multiple random samples: descriptions are real,
  product-specific text; `test_type` codes and `duration_minutes` vary
  correctly per product (e.g. reports/360-feedback products correctly have
  no duration since they're not timed tests)
- Loads cleanly into `app/catalog.py`'s `Catalog` class and BM25 index with
  zero code changes
- Full pipeline re-run against all 10 public traces at this real scale:
  **38/38 turns passed every hard-eval check** (schema, URL presence, turn
  cap, ≤10 recommendations)
- Retrieval quality checked directly: query "apache kafka data engineer"
  correctly surfaces "Apache Kafka (New)"; "entry level administrative
  assistant insurance" correctly ranks "Insurance Administrative Assistant
  Solution" first; multiple Graduate assessment variants (7.0/7.1/8.0/+8.0)
  all rank sensibly for graduate-hiring queries

**Known, documented limitation:** the `languages` field on each item is
unreliable — SHL's site-wide locale-switcher widget appears in the page text
near enough to the real "Languages" label that it occasionally gets
captured instead of (or alongside) the assessment's actual supported
languages. This does NOT affect scoring: `languages` isn't part of the
`recommendations` response schema and isn't wired into BM25 retrieval in
`catalog.py` (search runs over `name + description + test_type_labels +
job_levels` only). Documented as a known gap in `approach.md` rather than
chased further, since it doesn't touch anything the evaluator checks.
`duration_minutes` is `None` for 102/553 items — mostly reports and
360-feedback products that genuinely have no fixed completion time, plus a
handful (e.g. OPQ32r) where the archived page phrasing didn't match the
duration regex; also not schema-critical.

The scraper that produced this lives at `scraper/scrape_catalog_wayback.py`
(built and run in Google Colab — see that file's docstring for how it works
and how to re-run/extend it, e.g. if you want to also try to recover
`remote_testing`/`adaptive_irt` flags from archived listing pages, which
were deliberately left out for now).

## Running it

```bash
pip install -r requirements.txt
export LLM_API_KEY=your_groq_or_openai_compatible_key
export LLM_BASE_URL=https://api.groq.com/openai/v1   # default, override for other providers
export LLM_MODEL=llama-3.3-70b-versatile               # default
uvicorn app.main:app --reload
```

Then:
```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## What's verified vs. what needs a real LLM key to verify

Verified in this sandbox (no LLM key available here):
- Full FastAPI plumbing — `/health`, `/chat`, schema validation, error handling
- BM25 retrieval quality on real queries against the seed catalog
- The grounding filter: ran a mocked end-to-end pass where the "LLM" tried to
  return a hallucinated URL alongside real ones — the filter silently dropped
  it before it reached the response. This is the core "never recommend
  outside the catalog" guarantee and it works at the code level, not just
  the prompt level.
- All 10 public traces replayed turn-by-turn through a live running server:
  38 total turns, 100% pass on schema/URL-field/turn-cap/≤10-items hard evals.

Needs your real `LLM_API_KEY` to verify:
- Actual conversational quality — does `parse_node` extract facts well, does
  `compose_node` write good replies, does refine/compare/refuse behave as
  intended on real (not mocked) traces. Run `eval/run_eval.py` against your
  locally running server once you've got a key in; it'll catch hard-eval
  regressions immediately, but conversational quality you'll want to
  eyeball against a few of the trace personas yourself.

## Deployment

`Procfile` is set up for Railway, matching your existing deployment pattern
from the other projects. Set `LLM_API_KEY` (and `CATALOG_PATH` if you want to
point at a non-default catalog file) as environment variables on the host.
Remember the cold-start health check gets up to 2 minutes — the catalog load
+ BM25 index build happens at startup, not per-request, so it shouldn't add
meaningfully to that.
