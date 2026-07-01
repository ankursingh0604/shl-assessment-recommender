"""
Catalog store + retrieval.

Why BM25 instead of embeddings: the catalog is ~350-400 items, and queries
are keyword-dense ("Java", "Spring", "HIPAA", "contact centre agents") rather
than abstract paraphrases. BM25 gives exact, explainable, zero-cost recall on
that kind of query and needs no vector DB, no embedding API key, and no cold
start. The known weakness is pure semantic paraphrase ("people who handle
incoming money" not matching "Accounts Payable") — we partially cover that
gap upstream, in the LLM query-expansion step in graph.py, which turns vague
facts into multiple concrete keyword queries before they ever hit BM25.

Every recommendation the agent returns is required to come from a candidate
set this module produced — never from free LLM generation — which is what
actually enforces "never recommend anything outside the SHL catalog" (a
prompt instruction alone is not a guarantee; a programmatic filter is).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class CatalogEntry:
    name: str
    url: str
    test_type: list  # letter codes, e.g. ["K"]
    test_type_labels: list
    description: str
    duration_raw: str = ""
    languages_sample: str = ""
    job_levels: list = None

    @property
    def test_type_str(self) -> str:
        return ",".join(self.test_type)

    def to_recommendation_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type_str}

    def search_document(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.test_type_labels),
            " ".join(self.job_levels or []),
        ]
        return " ".join(p for p in parts if p)


class Catalog:
    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text())
        self.entries: list[CatalogEntry] = []
        for item in raw:
            self.entries.append(CatalogEntry(
                name=item["name"],
                url=item["url"],
                test_type=item.get("test_type", []),
                test_type_labels=item.get("test_type_labels", []),
                description=item.get("description", ""),
                duration_raw=item.get("duration_raw", ""),
                languages_sample=item.get("languages_sample", ""),
                job_levels=item.get("job_levels", []),
            ))
        self._by_url = {e.url: e for e in self.entries}
        self._by_name_lower = {e.name.lower(): e for e in self.entries}
        corpus = [_tokenize(e.search_document()) for e in self.entries]
        self._bm25 = BM25Okapi(corpus)

    def __len__(self):
        return len(self.entries)

    def search(self, query: str, top_k: int = 15) -> list[CatalogEntry]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.entries, scores), key=lambda x: x[1], reverse=True)
        return [e for e, s in ranked[:top_k] if s > 0]

    def multi_search(self, queries: list[str], top_k_each: int = 8, top_k_total: int = 20) -> list[CatalogEntry]:
        """Run several queries (e.g. one per extracted skill/constraint) and
        merge, deduplicated, preserving best-first order across all queries.
        Used so a single vague user message doesn't lose a skill's recall to
        whichever keyword happened to dominate a single combined query."""
        seen, merged = set(), []
        for q in queries:
            for e in self.search(q, top_k=top_k_each):
                if e.url not in seen:
                    seen.add(e.url)
                    merged.append(e)
        return merged[:top_k_total]

    def get_by_url(self, url: str) -> CatalogEntry | None:
        return self._by_url.get(url)

    def get_by_name(self, name: str) -> CatalogEntry | None:
        return self._by_name_lower.get(name.lower())

    def fuzzy_find(self, name_query: str) -> CatalogEntry | None:
        """Best-effort lookup for compare-mode, where the user names a product
        casually ('OPQ', 'GSA') rather than with its exact catalog title."""
        name_query_l = name_query.lower().strip()
        if name_query_l in self._by_name_lower:
            return self._by_name_lower[name_query_l]
        hits = self.search(name_query, top_k=1)
        return hits[0] if hits else None

    def find_names_mentioned(self, text: str) -> list[CatalogEntry]:
        """Scan arbitrary text (e.g. full conversation history) for catalog
        item names that appear verbatim, case-insensitive.

        Used as a code-level safety net for refine turns: if the agent
        already recommended something by name in an earlier turn, that
        item's URL should stay retrievable this turn even if this turn's
        generated search_queries happen to focus only on alternatives to it
        (a real failure mode observed in testing — the LLM correctly decided
        to keep an existing item, but it had already fallen out of the
        candidate pool, so the grounding filter stripped it). This doesn't
        replace the prompt-level instruction to search for it explicitly;
        it's a backstop for when that instruction isn't followed.
        """
        text_lower = text.lower()
        found = []
        for entry in self.entries:
            # Require reasonably long names to avoid short-name false
            # positives matching common English words inside sentences.
            if len(entry.name) >= 6 and entry.name.lower() in text_lower:
                found.append(entry)
        return found