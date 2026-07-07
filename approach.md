Approach — SHL Assessment Recommender
Architecture
A LangGraph state machine, not a single prompt, because the four required behaviors (clarify / recommend / refine /
compare) are fundamentally different operations, not variations in tone:
parse_node --(route)--> retrieve_node ------\
 --> compare_lookup_node >-- compose_node --> response
 --> noop (clarify/refuse)/
• parse_node classifies the turn into clarify / recommend / compare / refuse and extracts facts, search queries, and
compare targets as a structured Pydantic object.
•
retrieve_node / compare_lookup_node fetch candidates from the catalog based on that classification — never from
the LLM's own knowledge.
•
compose_node writes the reply and selects which candidate URLs to surface.
• A grounding filter runs after compose, outside the LLM: it intersects selected_urls with the actual candidate set
before anything becomes a recommendations entry. An LLM that names a URL it wasn't given gets silently corrected,
not trusted — this is the actual enforcement mechanism for "never recommend outside the catalog," not just a prompt
instruction.
Routing explicitly in code (rather than trusting one prompt to handle all four behaviors) means a reviewer can see the
decision point, and each path gets exactly the context it needs.
Retrieval
BM25 over the ~550-item catalog rather than embeddings: queries in this domain are keyword-dense ("Java", "Spring",
"HIPAA", "contact centre agents") rather than abstract paraphrases, so BM25 gives exact, cheap, zero-infra recall without a
vector DB or embedding API key. The known gap — pure semantic paraphrase ("people who handle incoming money" not
matching "Accounts Payable") — is partially covered upstream by having the parse stage turn vague facts into several
concrete keyword queries before they hit BM25, rather than one combined query.
multi_search runs one query per extracted skill/constraint and merges results round-robin (not sequential-fill) before
truncating to top_k_total. This was a real bug found during testing: a sequential merge let the first 2–3 skills in a multi-skill
query (e.g. "Java, Spring, SQL, AWS, Docker") consume the entire candidate budget, silently starving out skills mentioned
later in the list even when they had strong, relevant hits of their own.
fuzzy_find (used for compare-mode lookups like "OPQ" or "GSA") has a tiered fallback: exact name → prefix-word match
→ acronym match → BM25. Pure BM25 alone fails here because the tokenizer fuses names like "OPQ32r" into a single
token with zero overlap against a query for "OPQ" — there's no partial-token matching in BM25. The prefix-word tier also
needed a tiebreak toward names carrying a version code (OPQ32r) over shorter report-variant names sharing the same
prefix ("OPQ User Report"), since the catalog has ~20 OPQ-prefixed report variants and only one base instrument.
Grounding & Schema
Every /chat response is validated against a strict Pydantic schema before returning. Internal failures (LLM provider errors,
malformed JSON) never propagate as 500s — they degrade to a valid-schema apology reply with empty recommendations,
since an evaluator replaying turns needs consistent shape more than a technically-correct error code. The turn cap (8) is
enforced outside the graph as a hard protocol rule, not left to the LLM's judgment.
Evaluation
A local harness (eval/run_eval.py) replays the 10 public traces verbatim against a running /chat endpoint and checks:
• Hard evals: schema validity every turn, ≤10 recommendations, only catalog URLs, turn cap honored.
• Recall proxy: diffs the final shortlist returned against each trace's own final shortlist table. This is explicitly a sanity
signal, not the real score — the trace's shown shortlist isn't SHL's actual labeled ground truth, which isn't available to
us.
What Didn't Work / Debugging Story
Three real bugs surfaced only once evaluation moved from "schema passes" to actually measuring recall — schema
validity alone gave false confidence:
• Silent env var mismatch. parse_node/compose_node caught all exceptions and fell back to a generic "clarify"
response with no logging. Every single trace/turn returned zero recommendations, which looked like a retrieval or
prompt problem. Adding exception logging revealed a KeyError on the LLM API key — the underlying cause, invisible
until logging existed.
• Sequential-fill starvation in multi_search (described above), caught by testing round-robin interleaving directly
against the real catalog data and comparing candidate sets for multi-skill queries.
• Groq's free-tier daily token limit repeatedly interrupted eval runs mid-pass, producing false "zero recall" readings
that were actually quota exhaustion, not logic failures — distinguishable only by adding debug output showing raw
recommendation counts per turn and checking server-side logs for 429s.
The throughline: schema-level hard evals passing does not mean the system works — several of these bugs were fully
invisible until recall/behavior were actually measured, which is why building the recall-proxy check and turning on logging
early mattered more than any single retrieval tweak.
AI Tool Usage
Used Claude for: debugging (tracing the silent-failure bugs above via added logging), reviewing retrieval logic for
correctness, drafting/fixing the local eval harness, and deployment troubleshooting (Render env vars, Groq quota
diagnosis). Core agent design (the graph structure, the grounding-filter enforcement approach, the BM25-over-embeddings
tradeoff) and the retrieval bug fixes themselves were reasoned through and directed manually; AI tooling was used for
implementation speed and catching failure modes not yet tested for, not for the underlying design decisions.
Stack
FastAPI + LangGraph + Groq (Llama 3.3 70B, OpenAI-compatible API) + BM25 (rank_bm25) + Pydantic for structured LLM
output. Deployed on Render (free tier), kept warm via a 10-minute external health-check ping to avoid cold-start timeouts
during evaluation