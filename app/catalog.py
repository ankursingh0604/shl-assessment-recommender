"""
Catalog store + retrieval.
BM25 instead of embeddings: the catalog is ~350-400 items, and queries
are keyword-dense ("Java", "Spring", "HIPAA", "contact centre agents") rather
than abstract paraphrases.
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
        merge, deduplicated. Used so a single vague user message doesn't lose
        a skill's recall to whichever keyword happened to dominate a single
        combined query.
        """
        per_query_hits = [self.search(q, top_k=top_k_each) for q in queries]
        seen, merged = set(), []
        max_len = max((len(h) for h in per_query_hits), default=0)
        for rank in range(max_len):
            for hits in per_query_hits:
                if rank < len(hits):
                    e = hits[rank]
                    if e.url not in seen:
                        seen.add(e.url)
                        merged.append(e)
            if len(merged) >= top_k_total:
                break
        return merged[:top_k_total]

    def get_by_url(self, url: str) -> CatalogEntry | None:
        return self._by_url.get(url)

    def get_by_name(self, name: str) -> CatalogEntry | None:
        return self._by_name_lower.get(name.lower())
        name_query_l = name_query.lower().strip()
        if name_query_l in self._by_name_lower:
            return self._by_name_lower[name_query_l]

        prefix_hits = [
            (entry, next(w for w in entry.name.lower().split() if w.startswith(name_query_l)))
            for entry in self.entries
            if any(w.startswith(name_query_l) for w in entry.name.lower().split())
        ]
        if prefix_hits:
            # Prefer the matching word that carries a version/model code
            # (e.g. 'opq32r') over a bare acronym repeated in report-variant
            # names (e.g. 'OPQ User Report', 'OPQ Leadership Report' — all
            # of which also start with 'opq' but are variants, not the base
            # instrument). Catalogs like this consistently mark the actual
            # instrument with a trailing digit in its code; report variants
            # don't have one. Among remaining ties, prefer the shortest name.
            def _rank(pair):
                entry, matched_word = pair
                has_digit = any(ch.isdigit() for ch in matched_word)
                return (0 if has_digit else 1, len(entry.name))
            return min(prefix_hits, key=_rank)[0]

        stopwords = {"and", "of", "the", "for", "in", "a", "an"}
        for entry in self.entries:
            words = [w for w in re.findall(r"[a-zA-Z0-9]+", entry.name) if w.lower() not in stopwords]
            acronym = "".join(w[0] for w in words if w).lower()
            if acronym == name_query_l:
                return entry

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
            if len(entry.name) >= 6 and entry.name.lower() in text_lower:
                found.append(entry)
        return found