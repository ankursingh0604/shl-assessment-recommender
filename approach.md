# Approach Document — Conversational SHL Assessment Recommender

## Architecture

A LangGraph agent behind a stateless FastAPI service: `parse → route →
{retrieve | compare-lookup | noop} → compose`, with a programmatic grounding
filter between the LLM's output and the HTTP response.

**Why a graph, not one prompt.** Clarify, recommend, refine, compare, and
refuse are different *operations*, not different tones — clarify needs no
retrieval, recommend/refine needs catalog search, compare needs a lookup of
named items, refuse needs neither. Routing them as explicit graph edges means
each path gets exactly the context it needs, and "ask vs. retrieve vs.
answer vs. refuse" is a visible conditional edge in code, not something
buried in a long prompt.

**Why re-derive state every turn instead of patching it.** The API is
stateless — every call carries full history, nothing is stored. I considered
an incrementally-patched fact-state but rejected it: a misapplied patch
(e.g. failing to register "drop REST" against last turn's list) silently
corrupts every later turn, and correctness becomes order-dependent. Instead
`parse_node` re-reads the full transcript every turn and re-derives facts
from scratch — more tokens per call, bounded by the 8-turn cap, but
idempotent and much easier to reason about.

## Catalog acquisition

The live catalog listing page was redirecting to a generic overview page at
scrape time (confirmed in a real browser — not bot-detection). I used the
Wayback Machine's CDX API to discover every URL ever archived under the
catalog's detail-page path and pulled each item's most recent working
snapshot directly, bypassing the broken listing page entirely. Result:
**553 items**, 0 missing description, 0 missing test_type, verified to load
cleanly and re-pass all 10 public traces (38/38 turns, all hard evals).
Known gaps, neither scoring-relevant: `remote_testing`/`adaptive_irt` flags
default to False (only visible on the listing page, not scraped); a handful
of items have an unreliable `languages` field where a site-wide nav widget
occasionally gets captured instead of the real per-item list.

## Retrieval: BM25, not embeddings

Queries here are keyword-dense ("Java," "HIPAA," "contact centre agents"),
not abstract paraphrase, so BM25 over `name + description + test-type labels
+ job levels` gives exact, free, deterministic recall with no vector DB or
embedding cost. Its known weakness — missing pure paraphrase like "people who
handle incoming money" vs. "Accounts Payable" — is mitigated upstream:
`parse_node` extracts 2–5 concrete keyword queries from the accumulated
facts rather than searching the user's raw sentence, pushing the semantic
translation to the LLM and keeping retrieval itself simple and explainable.

## Grounding is enforced in code, not just the prompt

The compose prompt restricts the LLM to `CANDIDATES`, but a prompt
instruction isn't a guarantee. After compose returns `selected_urls`, the
server intersects that list against the actual retrieved candidates before
they become the response's `recommendations`. Verified directly: a test had
the model return one real URL plus one invented one, and the invented one
was silently dropped before the response left the server. This — not the
prompt wording — is what actually enforces "every URL must come from the
catalog."

## Two real bugs found through live testing

Both produced valid, schema-compliant responses with subtly wrong content —
neither would have been caught by schema checks or mocked tests alone.

**Bug 1 — near-duplicate names caused a coverage miss.** Asked for a
graduate battery covering cognitive, personality, and situational judgement,
the shortlist was missing cognitive coverage entirely. Root cause: two
near-identically-named catalog products — "Graduate 8.0 Job Focused
Assessment" (`P,B`) and "Graduate + 8.0 Job Focused Assessment" (`A,B,P`) —
differ by one character; compose picked the wrong one, apparently by name
similarity rather than checking `test_type`. Fix: added an explicit
coverage self-check to the compose prompt — verify each requested dimension
against candidates' `test_type` fields, not names, with this pair flagged as
a known trap. Verified fixed by re-running the identical query.

**Bug 2 — refine turns could silently drop an already-recommended item.**
Asked to swap one item in an existing shortlist for something "shorter," the
agent correctly declined (nothing shorter existed with equal coverage) but
the response only contained 2 of the original 3 items, and the dropped item
was misnamed in the reply text. Root cause: retrieval for a given turn only
returns what that turn's search queries surface; a query about *alternatives*
to an item doesn't necessarily re-surface the item itself, so it silently
fell out of the candidate pool — and the grounding filter (correctly) won't
let compose recommend something outside that pool, even if it wants to keep
it. Fixed at two layers: (1) prompt-level, instructing parse to always query
for an item under discussion, not just its alternatives; (2) code-level —
added `Catalog.find_names_mentioned()`, which scans conversation history for
catalog names already mentioned and guarantees they stay retrievable
regardless of what the LLM's queries produce. Same philosophy as the URL
grounding filter: don't rely on a model reliably following an instruction
when a deterministic guarantee is available. Verified fixed: re-ran the
identical conversation, all 3 items returned correctly, with the correct
name.

## What I'd improve with more time

A local-embedding + BM25 hybrid (fused via reciprocal rank fusion) to
recover pure-paraphrase queries BM25 structurally can't match; an LLM
rerank pass over BM25's top-K to improve precision on borderline candidates;
a larger set of hand-written adversarial refuse-path probes beyond the 10
public traces.

## AI tool usage

Built with Claude (agentic coding assistance) for scaffolding, the retrieval
module, and test harnesses. I drove the architecture decisions above (graph
structure, BM25 vs. embeddings, re-derive-vs-patch state, where grounding is
enforced) and personally ran the live tests that found both bugs above,
verifying each fix before treating it as done.