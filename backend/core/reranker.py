import asyncio
import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import (  # pyright: ignore[reportMissingTypeStubs]
    TextCrossEncoder,
)
from langchain_core.documents import Document

from backend.core.config import settings
from backend.core.logging import app_logger


@lru_cache(maxsize=1)
def _get_reranker() -> TextCrossEncoder:
    app_logger.info("Loading reranker model: %s", settings.rerank.model_name)
    return TextCrossEncoder(model_name=settings.rerank.model_name)


async def arerank(
    query: str,
    scored_docs: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """Rerank retrieved candidates by cross-encoder score and return top_k."""
    if not scored_docs:
        return scored_docs

    docs = [doc for doc, _ in scored_docs]
    texts = [doc.page_content for doc in docs]
    batch_size = settings.rerank.batch_size

    scores: list[float] = await asyncio.to_thread(
        lambda: list(
            _get_reranker().rerank(query=query, documents=texts, batch_size=batch_size)
        )
    )

    normalized = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    paired = sorted(zip(docs, normalized), key=lambda x: x[1], reverse=True)
    return paired[:top_k]
