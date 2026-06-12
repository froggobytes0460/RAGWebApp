import asyncio
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal, Self, cast

from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.conversions.common_types import UpdateResult
from qdrant_client.models import (
    DatetimeIndexParams,
    DatetimeIndexType,
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    KeywordIndexParams,
    KeywordIndexType,
    MatchValue,
    VectorParams,
)

from backend.core.config import settings
from backend.core.ingest import StrictMetadata


@lru_cache(maxsize=1)
def _get_huggingface_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.vector_store.embedding_model,
        encode_kwargs={
            "normalize_embeddings": settings.vector_store.normalize_embeddings
        },
    )


class VectorStore(BaseModel):
    client: QdrantClient
    async_client: AsyncQdrantClient

    k: int = settings.search.top_k
    collection_name: str = settings.vector_store.collection_name
    vector_size: int = settings.vector_store.vector_size
    ttl: int = settings.vector_store.ttl

    _vector_store: QdrantVectorStore | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_settings(cls) -> Self:
        vectorstore_url_or_path = settings.vector_store.url_or_path
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        if isinstance(vectorstore_url_or_path, Path):
            client = QdrantClient(path=str(vectorstore_url_or_path))
            async_client = AsyncQdrantClient(path=str(vectorstore_url_or_path))
        else:
            client = QdrantClient(url=str(vectorstore_url_or_path), api_key=api_key)
            async_client = AsyncQdrantClient(
                url=str(vectorstore_url_or_path), api_key=api_key
            )
        return cls(client=client, async_client=async_client)

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=_get_huggingface_embeddings(),
            )
        return self._vector_store

    async def ainit_collection(
        self,
    ) -> dict[Literal["session_id", "uploaded_at"], UpdateResult]:
        exists: bool = await self.async_client.collection_exists(
            collection_name=self.collection_name
        )
        if not exists:
            if not await self.async_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            ):
                raise RuntimeError("Unable to initialize Qdrant collection.")

        session_id_res, uploaded_at_res = await asyncio.gather(
            self.async_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.session_id",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            ),
            self.async_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.uploaded_at",
                field_schema=DatetimeIndexParams(type=DatetimeIndexType.DATETIME),
            ),
        )

        return {
            "session_id": session_id_res,
            "uploaded_at": uploaded_at_res,
        }

    async def ainsert_docs(
        self, documents: list[Document], session_id: str
    ) -> list[str]:
        curr_time = datetime.now(tz=timezone.utc).isoformat()
        for doc in documents:
            doc_metadata: StrictMetadata = cast(StrictMetadata, doc.metadata)
            doc_metadata["session_id"] = session_id
            doc_metadata["uploaded_at"] = curr_time

        ids: list[str] = await self.vector_store.aadd_documents(documents)

        if len(ids) != len(documents):
            raise RuntimeError("Mismatch between inserted docs and returned IDs")

        return ids

    def get_retriever(
        self, session_id: str, k: int | None = None
    ) -> VectorStoreRetriever:
        search_kwargs: dict[str, int | Filter | float] = {
            "k": k or self.k,
            "filter": Filter(
                must=[
                    FieldCondition(
                        key="metadata.session_id", match=MatchValue(value=session_id)
                    )
                ]
            ),
        }

        if settings.search.search_type == "mmr":
            search_kwargs["fetch_k"] = (k or self.k) * 4
            search_kwargs["lambda_mult"] = settings.search.lambda_mult

        elif settings.search.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = settings.search.score_threshold

        return self.vector_store.as_retriever(
            search_type=settings.search.search_type,
            search_kwargs=search_kwargs,
        )

    async def adelete_session(self, session_id: str) -> UpdateResult:
        return await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.session_id", match=MatchValue(value=session_id)
                    )
                ]
            ),
        )

    async def aclean_up_stale_vectors(self) -> UpdateResult:
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=settings.vector_store.ttl
        )
        return await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.uploaded_at",
                        range=DatetimeRange(lt=cutoff),
                    )
                ]
            ),
        )

    async def aclose(self) -> None:
        await self.async_client.close()
