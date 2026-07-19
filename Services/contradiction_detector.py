"""LLM-based pairwise contradiction detection across witness testimonies.

Given a set of search-result chunks for a query, group by (witness,
inquiry), build cross-group pairs, and ask Claude Haiku 4.5 whether each
pair contradicts. Grouping by (witness, inquiry) — not name alone — means a
witness who testified in BOTH inquiries under the same name (Ismay, Stanley
Lord, Fleet...) is compared against their own other-inquiry testimony, the
same way differently-named cross-inquiry witnesses (Lightoller) always were.

Pair checks run concurrently on a small thread pool; verdicts are cached via
the pluggable ContradictionCache so repeat queries hit cache instead of the
LLM.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import anthropic
from pydantic import BaseModel, Field

from Services.british_witness_index import canonical_witness_name
from Services.contradiction_cache import ContradictionCache, make_cache
from Services.embeddings import EmbeddedChunk

logger = logging.getLogger(__name__)


MODEL_ID = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You analyze pairs of Titanic witness statements and detect factual "
    "contradictions. Reply only with valid JSON matching the schema. A "
    "contradiction requires both statements to make claims on the same "
    "specific fact (numbers, times, sequence, identity) that cannot both be "
    "true. Vague, general, or non-overlapping statements are NOT "
    "contradictions."
)


class ContradictionVerdict(BaseModel):
    contradicts: bool
    claim_a: str = Field(..., description="Short paraphrase of A's specific claim")
    claim_b: str = Field(..., description="Short paraphrase of B's specific claim")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(
        ..., description="One sentence on why these conflict (or why they don't)"
    )


@dataclass
class Contradiction:
    witness_a: str
    witness_b: str
    chunk_a: str
    chunk_b: str
    claim_a: str
    claim_b: str
    confidence: float
    explanation: str
    contradicts: bool = True
    source_a: str = ""       # "us_inquiry" / "british_inquiry"
    source_b: str = ""
    page_a: int = 0          # printed/transcript page of the quoted chunk
    page_b: int = 0
    role_a: str = ""         # e.g. "2nd Officer, Titanic"
    role_b: str = ""
    same_person: bool = False  # same witness across the two inquiries


def _chunk_id(chunk: EmbeddedChunk) -> str:
    """Stable short hash of witness + content — used for cache keying."""
    src = (chunk.chunk.witness_name + "::" + chunk.chunk.content).encode("utf-8")
    return hashlib.sha256(src).hexdigest()[:16]


def _same_person(key_a: Tuple[str, str], key_b: Tuple[str, str]) -> bool:
    """True when the two (name, source) groups are the same human testifying
    in the two different inquiries — via exact name match or the
    British→US canonical alias map."""
    (name_a, source_a), (name_b, source_b) = key_a, key_b
    if source_a == source_b:
        return False
    return canonical_witness_name(name_a) == canonical_witness_name(name_b)


class ContradictionDetector:
    # Cap on LLM pair-checks per query: 45 pairs ≈ 10 distinct witnesses.
    # Checks run on max_workers threads, so worst case stays under ~10s.
    MAX_PAIRS = 45

    def __init__(
        self,
        cache: Optional[ContradictionCache] = None,
        client: Optional[anthropic.Anthropic] = None,
        model: str = MODEL_ID,
        max_workers: int = 8,
    ):
        self.cache = cache if cache is not None else make_cache()
        # 30s timeout — default is 10 minutes, which would hang /search/contradictions
        # under a stuck call. Override here, not at client level, so injected clients
        # keep their own settings.
        self.client = client if client is not None else anthropic.Anthropic(timeout=30.0)
        self.model = model
        self.max_workers = max_workers

    def detect(self, chunks: List[EmbeddedChunk], query: str) -> List[Contradiction]:
        """Find contradictions across the given chunks for this query.

        Groups chunks by (witness_name, source_type) and compares the
        highest-ranked chunk of each group against every other group's.
        The caller is expected to pass chunks already ranked by search
        relevance (first per group = best).
        """
        groups: Dict[Tuple[str, str], List[EmbeddedChunk]] = {}
        order: List[Tuple[str, str]] = []  # first-appearance = rank order
        for c in chunks:
            key = (c.chunk.witness_name, c.chunk.metadata.source_type)
            if key not in groups:
                order.append(key)
            groups.setdefault(key, []).append(c)

        pairs = [
            (order[i], order[j])
            for i in range(len(order))
            for j in range(i + 1, len(order))
        ]
        if len(pairs) > self.MAX_PAIRS:
            logger.info("Truncating %d candidate pairs to %d", len(pairs), self.MAX_PAIRS)
            pairs = pairs[: self.MAX_PAIRS]
        if not pairs:
            return []

        def check(pair: Tuple[Tuple[str, str], Tuple[str, str]]) -> Optional[ContradictionVerdict]:
            a, b = groups[pair[0]][0], groups[pair[1]][0]
            try:
                return self._check_pair(a, b, query)
            except Exception:
                logger.exception(
                    "Pair check failed: %s vs %s", pair[0][0], pair[1][0]
                )
                return None  # one bad call must not sink the whole request

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pairs))) as pool:
            verdicts = list(pool.map(check, pairs))

        contradictions: List[Contradiction] = []
        for (key_a, key_b), verdict in zip(pairs, verdicts):
            if verdict is None or not verdict.contradicts:
                continue
            a, b = groups[key_a][0], groups[key_b][0]
            contradictions.append(
                Contradiction(
                    witness_a=key_a[0],
                    witness_b=key_b[0],
                    chunk_a=a.chunk.content,
                    chunk_b=b.chunk.content,
                    claim_a=verdict.claim_a,
                    claim_b=verdict.claim_b,
                    confidence=verdict.confidence,
                    explanation=verdict.explanation,
                    source_a=key_a[1],
                    source_b=key_b[1],
                    page_a=a.chunk.metadata.page_number,
                    page_b=b.chunk.metadata.page_number,
                    role_a=a.chunk.metadata.role,
                    role_b=b.chunk.metadata.role,
                    same_person=_same_person(key_a, key_b),
                )
            )

        contradictions.sort(key=lambda c: c.confidence, reverse=True)
        return contradictions

    def _cache_key(self, a: EmbeddedChunk, b: EmbeddedChunk, query: str) -> str:
        ids = sorted([_chunk_id(a), _chunk_id(b)])
        q = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"{ids[0]}:{ids[1]}:{q}"

    def _check_pair(
        self, a: EmbeddedChunk, b: EmbeddedChunk, query: str
    ) -> ContradictionVerdict:
        key = self._cache_key(a, b, query)
        cached = self.cache.get(key)
        if cached is not None:
            return ContradictionVerdict(**cached)

        user_prompt = (
            f'Query: "{query}"\n'
            f'Witness A ({a.chunk.witness_name}): "{a.chunk.content}"\n'
            f'Witness B ({b.chunk.witness_name}): "{b.chunk.content}"'
        )

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=ContradictionVerdict,
        )
        verdict: ContradictionVerdict = response.parsed_output
        self.cache.put(key, verdict.model_dump())
        return verdict


def _cli() -> None:
    """python -m Services.contradiction_detector "<query>"

    Pulls top results from the existing search engine and runs detect() on
    them. Requires ANTHROPIC_API_KEY, plus OPENAI_API_KEY and Pinecone config
    so the search pipeline can run.
    """
    import json

    if len(sys.argv) < 2:
        print(
            'Usage: python -m Services.contradiction_detector "<query>"',
            file=sys.stderr,
        )
        sys.exit(2)

    query = sys.argv[1]

    from dotenv import load_dotenv
    load_dotenv()

    from Services.embeddings import EmbeddingService
    from Services.vector_storage import PineconeVectorStore
    from Services.semantic_search import SemanticSearchEngine, SearchQuery

    embedding_service = EmbeddingService()
    vector_store = PineconeVectorStore()
    search = SemanticSearchEngine(
        embedding_service,
        vector_store,
        default_top_k=8,
        similarity_threshold=0.4,
    )

    results = search.search(
        SearchQuery(text=query, top_k=8, similarity_threshold=0.4)
    )
    if not results:
        print(json.dumps(
            {"query": query, "contradictions": [], "note": "no search results"},
            indent=2,
        ))
        return

    chunks = [r.chunk for r in results]

    detector = ContradictionDetector()
    contradictions = detector.detect(chunks, query)

    output = {
        "query": query,
        "witnesses_compared": sorted({c.chunk.witness_name for c in chunks}),
        "contradictions": [asdict(c) for c in contradictions],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    _cli()