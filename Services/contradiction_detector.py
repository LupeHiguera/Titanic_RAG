"""LLM-based pairwise contradiction detection across witness testimonies.

Given a set of search-result chunks for a query, group by witness, build
cross-witness pairs, and ask Claude Haiku 4.5 whether each pair contradicts.
Verdicts are cached via the pluggable ContradictionCache so repeat queries
on the same witness pairs hit cache instead of the LLM.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import anthropic
from pydantic import BaseModel, Field

from Services.contradiction_cache import ContradictionCache, make_cache
from Services.embeddings import EmbeddedChunk


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


def _chunk_id(chunk: EmbeddedChunk) -> str:
    """Stable short hash of witness + content — used for cache keying."""
    src = (chunk.chunk.witness_name + "::" + chunk.chunk.content).encode("utf-8")
    return hashlib.sha256(src).hexdigest()[:16]


class ContradictionDetector:
    def __init__(
        self,
        cache: Optional[ContradictionCache] = None,
        client: Optional[anthropic.Anthropic] = None,
        model: str = MODEL_ID,
    ):
        self.cache = cache if cache is not None else make_cache()
        # 30s timeout — default is 10 minutes, which would hang /search/contradictions
        # under a stuck call. Override here, not at client level, so injected clients
        # keep their own settings.
        self.client = client if client is not None else anthropic.Anthropic(timeout=30.0)
        self.model = model

    def detect(self, chunks: List[EmbeddedChunk], query: str) -> List[Contradiction]:
        """Find contradictions across the given chunks for this query.

        Compares the highest-ranked chunk from each witness against every other
        witness's highest-ranked chunk. The caller is expected to pass chunks
        already ranked by search relevance (first per witness = best).
        """
        by_witness: dict[str, list[EmbeddedChunk]] = {}
        for c in chunks:
            by_witness.setdefault(c.chunk.witness_name, []).append(c)

        witnesses = list(by_witness)
        contradictions: List[Contradiction] = []

        for i, w_a in enumerate(witnesses):
            for w_b in witnesses[i + 1:]:
                a, b = by_witness[w_a][0], by_witness[w_b][0]
                verdict = self._check_pair(a, b, query)
                if verdict.contradicts:
                    contradictions.append(
                        Contradiction(
                            witness_a=w_a,
                            witness_b=w_b,
                            chunk_a=a.chunk.content,
                            chunk_b=b.chunk.content,
                            claim_a=verdict.claim_a,
                            claim_b=verdict.claim_b,
                            confidence=verdict.confidence,
                            explanation=verdict.explanation,
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