# pyright: reportAny=false
# pyright: reportExplicitAny=false

import asyncio
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, cast

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, ConfigDict, PrivateAttr
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import models
from qdrant_client.conversions import common_types

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
    """Vector store interface for RAG web application."""

    client: QdrantClient
    async_client: AsyncQdrantClient | None

    k: int = settings.search.top_k
    collection_name: str = settings.vector_store.collection_name
    vector_size: int = settings.vector_store.vector_size
    ttl: int = settings.vector_store.ttl

    _vector_store: QdrantVectorStore | None = PrivateAttr(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_settings(cls) -> Self:
        vectorstore_url_or_path = settings.vector_store.url_or_path
        api_key = (
            settings.vector_store.api_key.get_secret_value()
            if settings.vector_store.api_key
            else None
        )
        if isinstance(vectorstore_url_or_path, Path):
            client = QdrantClient(path=str(vectorstore_url_or_path))
            async_client = None
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

    async def _run_async(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if self.async_client:
            method = getattr(self.async_client, method_name)
            return await method(*args, **kwargs)

        method = getattr(self.client, method_name)
        return await asyncio.to_thread(method, *args, **kwargs)

    async def ainit_collection(
        self,
    ) -> dict[Literal["session_id", "uploaded_at"], models.UpdateResult]:
        exists: bool = await self._run_async(
            method_name="collection_exists", collection_name=self.collection_name
        )

        if not exists:
            created: bool = await self._run_async(
                method_name="create_collection",
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )
            if not created:
                raise RuntimeError("Unable to initialize Qdrant collection.")

        session_id_res, uploaded_at_res = await asyncio.gather(
            self._run_async(
                method_name="create_payload_index",
                collection_name=self.collection_name,
                field_name="metadata.session_id",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD
                ),
            ),
            self._run_async(
                method_name="create_payload_index",
                collection_name=self.collection_name,
                field_name="metadata.uploaded_at",
                field_schema=models.DatetimeIndexParams(
                    type=models.DatetimeIndexType.DATETIME
                ),
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

    async def asearch_with_scores(
        self, query: str, session_id: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        top_k = k or self.k
        session_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        )

        if settings.search.search_type == "mmr":
            fetch_k = top_k * 4
            lambda_mult = settings.search.lambda_mult
            docs: list[Document] = (
                await self.vector_store.amax_marginal_relevance_search(
                    query=query,
                    k=top_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=session_filter,
                )
            )
            scored: list[tuple[Document, float]] = (
                await self.vector_store.asimilarity_search_with_relevance_scores(
                    query=query,
                    k=fetch_k,
                    filter=session_filter,
                )
            )
            score_map = {d.page_content: s for d, s in scored}
            return [(doc, score_map.get(doc.page_content, 0.0)) for doc in docs]

        results: list[tuple[Document, float]] = (
            await self.vector_store.asimilarity_search_with_relevance_scores(
                query=query,
                k=top_k,
                filter=session_filter,
            )
        )

        if settings.search.search_type == "similarity_score_threshold":
            results = [
                (doc, score)
                for doc, score in results
                if score >= settings.search.score_threshold
            ]

        return results

    async def adelete_session(self, session_id: str) -> models.UpdateResult:
        return await self._run_async(
            method_name="delete",
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
        )

    async def adelete_document(
        self, session_id: str, filename: str
    ) -> models.UpdateResult:
        return await self._run_async(
            method_name="delete",
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    ),
                    models.FieldCondition(
                        key="metadata.filename", match=models.MatchValue(value=filename)
                    ),
                ]
            ),
        )

    async def aclean_up_stale_vectors(self) -> models.UpdateResult:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            seconds=settings.vector_store.ttl
        )
        return await self._run_async(
            method_name="delete",
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.uploaded_at",
                        range=models.DatetimeRange(lt=cutoff),
                    )
                ]
            ),
        )

    async def alist_documents(self, session_id: str) -> list[dict[str, str]]:
        response: tuple[list[models.Record], common_types.PointId | None] = (
            await self._run_async(
                method_name="scroll",
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.session_id",
                            match=models.MatchValue(value=session_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=models.PayloadSelectorInclude(
                    include=["metadata.filename", "metadata.uploaded_at"]
                ),
                with_vectors=False,
            )
        )

        records, _ = response
        unique_files: dict[str, str] = {}

        for record in records:
            payload = record.payload or {}
            meta = payload.get("metadata", {})
            if isinstance(meta, dict):
                filename = cast(
                    Any,
                    meta.get("filename"),  # pyright: ignore[reportUnknownMemberType]
                )
                uploaded_at = cast(
                    Any,
                    meta.get("uploaded_at"),  # pyright: ignore[reportUnknownMemberType]
                )
                if isinstance(filename, str) and isinstance(uploaded_at, str):
                    unique_files[filename] = uploaded_at

        return [
            {"filename": fname, "uploaded_at": uploaded}
            for fname, uploaded in unique_files.items()
        ]

    async def aclose(self) -> None:
        await self._run_async(method_name="close")
