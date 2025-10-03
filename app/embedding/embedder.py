from typing import List
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from typing import List
from app.core.config import settings
from app.core.aws_secrets_service import require_openai_api_key
import asyncio

_embedding_cache: dict[str, list[float]] = {}

def get_embedder() -> OpenAIEmbeddings:
    """Lazy initialization of OpenAI embedder"""
    openai_key = require_openai_api_key()
    return OpenAIEmbeddings(model=settings.embed_model, openai_api_key=openai_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=2.0))
def _embed_batch(batch: List[str]) -> List[list[float]]:
    """
    Synchronous embedding with retry.
    """
    embedder = get_embedder()  # Lazy initialization
    return embedder.embed_documents(batch)


async def embed_column_names_batched_async(names: List[str], batch_size: int = 50) -> List[List[float]]:
    """
    Async embedding with batching, retry, and caching.
    """
    results = []

    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        uncached = [name for name in batch if name not in _embedding_cache]

        # Embed only uncached
        if uncached:
            embedded = await asyncio.to_thread(_embed_batch, uncached)
            for name, vec in zip(uncached, embedded):
                _embedding_cache[name] = vec

        # Retrieve all from cache
        results.extend([_embedding_cache[name] for name in batch])

    return results
