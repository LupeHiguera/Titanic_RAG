import numpy as np
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import os

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

from Services.chunking import WitnessChunk


@dataclass
class EmbeddedChunk:
    chunk: WitnessChunk
    embedding: np.ndarray


class EmbeddingService:
    def __init__(self, provider: str = "openai", model: str = "text-embedding-3-large", api_key: Optional[str] = None, dimensions: int = 1024):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._cache: Dict[str, np.ndarray] = {}
        
        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key is required but not provided")
            if OpenAI is None:
                raise ImportError("openai package is required but not installed")
            self.client = OpenAI(api_key=self.api_key)
    
    def embed_chunk(self, chunk: WitnessChunk) -> EmbeddedChunk:
        """Embed a single chunk with rate limiting and caching."""
        if not chunk.content or chunk.content.strip() == "":
            raise ValueError("Cannot embed empty content")
        
        # Check cache first
        cache_key = self._get_cache_key(chunk.content)
        cached_embedding = self._get_from_cache(cache_key)
        if cached_embedding is not None:
            return EmbeddedChunk(chunk=chunk, embedding=cached_embedding)
        
        # Get embedding from provider
        embedding = self._get_embedding_with_retry(chunk.content)
        
        # Store in cache
        self._store_in_cache(cache_key, embedding)
        
        return EmbeddedChunk(chunk=chunk, embedding=embedding)
    
    def embed_batch(self, chunks: List[WitnessChunk]) -> List[EmbeddedChunk]:
        """Embed multiple chunks via batched OpenAI calls.

        Cached chunks are served from memory; uncached chunks are grouped
        into 100-input batches and embedded in a single API call per batch.
        Order is preserved.
        """
        if not chunks:
            return []

        results: List[Optional[EmbeddedChunk]] = [None] * len(chunks)
        to_embed: List[Tuple[int, WitnessChunk]] = []

        for i, chunk in enumerate(chunks):
            if not chunk.content or chunk.content.strip() == "":
                raise ValueError("Cannot embed empty content")
            cached = self._get_from_cache(self._get_cache_key(chunk.content))
            if cached is not None:
                results[i] = EmbeddedChunk(chunk=chunk, embedding=cached)
            else:
                to_embed.append((i, chunk))

        for batch_start in range(0, len(to_embed), 100):
            batch = to_embed[batch_start:batch_start + 100]
            texts = [chunk.content for _, chunk in batch]
            embeddings = self._get_embeddings_batch_with_retry(texts)
            for (orig_i, chunk), emb in zip(batch, embeddings):
                self._store_in_cache(self._get_cache_key(chunk.content), emb)
                results[orig_i] = EmbeddedChunk(chunk=chunk, embedding=emb)

        return results  # all slots populated above
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed raw text (used by search engine)."""
        if not text or text.strip() == "":
            raise ValueError("Cannot embed empty text")
        
        cache_key = self._get_cache_key(text)
        cached_embedding = self._get_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        embedding = self._get_embedding_with_retry(text)
        self._store_in_cache(cache_key, embedding)
        
        return embedding
    
    def _get_embedding_with_retry(self, text: str, max_retries: int = 3) -> np.ndarray:
        """Get embedding for a single text with retry on rate limits."""
        if self.provider != "openai":
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    dimensions=self.dimensions,
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep((2 ** attempt) + 1)
                    continue
                raise

    def _get_embeddings_batch_with_retry(self, texts: List[str], max_retries: int = 3) -> List[np.ndarray]:
        """Get embeddings for a batch of texts in a single API call, with retry."""
        if self.provider != "openai":
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.dimensions,
                )
                return [np.array(d.embedding) for d in response.data]
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep((2 ** attempt) + 1)
                    continue
                raise
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        similarity = dot_product / norms
        return float(similarity)
    
    def find_similar_chunks(self, query_embedding: np.ndarray, embedded_chunks: List[EmbeddedChunk], 
                           top_k: int = 5, similarity_threshold: float = 0.5) -> List[Tuple[EmbeddedChunk, float]]:
        """Find similar chunks to a query embedding."""
        similarities = []
        
        for embedded_chunk in embedded_chunks:
            similarity = self.cosine_similarity(query_embedding, embedded_chunk.embedding)
            if similarity >= similarity_threshold:
                similarities.append((embedded_chunk, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{self.dimensions}:{text}".encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        return self._cache.get(cache_key)
    
    def _store_in_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        self._cache[cache_key] = embedding
    
    def _split_into_batches(self, items: List[Any], batch_size: int = 100) -> List[List[Any]]:
        """Split list into batches."""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches