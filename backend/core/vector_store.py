import asyncio
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self, cast
import uuid

from fastembed.common.types import NumpyArray
from fastembed.text import TextEmbedding
from langchain_core.documents import Document
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from backend.core.config import VectorStoreSettings, settings
from backend.core.ingest import StrictMetadata
from backend.core.logging import app_logger


@lru_cache(maxsize=1)
def _get_fastembed_embeddings() -> TextEmbedding:
    app_logger.info(
        "Loading FastEmbed model: %s", settings.vector_store.embedding_model
    )
    return TextEmbedding(model_name=settings.vector_store.embedding_model)


def _embed(texts: list[str]) -> Iterable[NumpyArray]:
    model = _get_fastembed_embeddings()
    return model.embed(documents=texts)


class VectorStore:
    """Vector store interface for RAG web application."""

    _client: AsyncQdrantClient
    k: int
    vector_store_settings: VectorStoreSettings

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        k: int = settings.search.top_k,
        vector_store_settings: VectorStoreSettings,
    ) -> None:
        self.vector_store_settings = vector_store_settings
        self._client = client
        self.k = k

    @classmethod
    def from_settings(cls) -> Self:
        url_or_path = settings.vector_store.url_or_path
        api_key = (
            settings.vector_store.api_key.get_secret_value()
            if settings.vector_store.api_key
            else None
        )
        if isinstance(url_or_path, Path):
            client = AsyncQdrantClient(path=str(url_or_path))
        else:
            client = AsyncQdrantClient(
                url=str(url_or_path),
                api_key=api_key,
                prefer_grpc=settings.vector_store.prefer_qdrant_grpc,
            )
        return cls(
            client, k=settings.search.top_k, vector_store_settings=settings.vector_store
        )

    async def ainit_collection(
        self,
    ) -> dict[Literal["session_id", "uploaded_at"], models.UpdateResult]:
        app_logger.info(
            "Initializing Qdrant collection: %s",
            self.vector_store_settings.collection_name,
        )
        exists: bool = await self._client.collection_exists(
            collection_name=self.vector_store_settings.collection_name
        )

        if not exists:
            created: bool = await self._client.create_collection(
                collection_name=self.vector_store_settings.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_store_settings.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            if not created:
                raise RuntimeError("Unable to initialize Qdrant collection.")
            app_logger.info(
                "Created new Qdrant collection: %s",
                self.vector_store_settings.collection_name,
            )

        session_id_res, uploaded_at_res = await asyncio.gather(
            self._client.create_payload_index(
                collection_name=self.vector_store_settings.collection_name,
                field_name="metadata.session_id",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD
                ),
            ),
            self._client.create_payload_index(
                collection_name=self.vector_store_settings.collection_name,
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
        app_logger.debug("Inserting %d docs (session: %s)", len(documents), session_id)
        curr_time = datetime.now(tz=timezone.utc).isoformat()
        for doc in documents:
            doc_metadata: StrictMetadata = doc.metadata  # type: ignore[assignment]
            doc_metadata["session_id"] = session_id
            doc_metadata["uploaded_at"] = curr_time

        texts = [doc.page_content for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        _ = await self._client.upsert(
            collection_name=self.vector_store_settings.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,  # pyright: ignore[reportArgumentType]
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                for point_id, vector, doc in zip(
                    ids,
                    (await asyncio.to_thread(_embed, texts)),
                    documents,
                )
            ],
        )

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

        [query_vector] = await asyncio.to_thread(_embed, [query])

        if settings.search.search_type == "mmr":
            return await self._mmr_search(
                query_vector=query_vector,
                session_filter=session_filter,
                top_k=top_k,
            )

        response = await self._client.query_points(
            collection_name=self.vector_store_settings.collection_name,
            query=query_vector,
            query_filter=session_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=(
                settings.search.score_threshold
                if settings.search.search_type == "similarity_score_threshold"
                else None
            ),
        )

        return [
            (self._point_to_document(point=hit), hit.score) for hit in response.points
        ]

    async def _mmr_search(
        self,
        query_vector: NumpyArray,
        session_filter: models.Filter,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        fetch_k = top_k * 4
        lambda_mult = settings.search.lambda_mult

        mmr_response = await self._client.query_points(
            collection_name=self.vector_store_settings.collection_name,
            query=query_vector,
            query_filter=session_filter,
            limit=fetch_k,
            with_payload=True,
            with_vectors=True,
        )
        hits: list[models.ScoredPoint] = mmr_response.points

        if not hits:
            return []

        n = len(hits)
        candidate_vecs = np.array([hit.vector for hit in hits], dtype=float)
        relevance_scores = np.array([hit.score for hit in hits], dtype=float)

        max_sim_to_selected = np.zeros(n, dtype=float)
        remaining: set[int] = set(range(n))
        selected_indices: list[int] = []

        while remaining and len(selected_indices) < top_k:
            rem = list(remaining)
            mmr_scores = (
                lambda_mult * relevance_scores[rem]
                - (1 - lambda_mult) * max_sim_to_selected[rem]
            )
            best = rem[int(np.argmax(mmr_scores))]
            selected_indices.append(best)
            remaining.remove(best)

            rest = list(remaining)
            sims = cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                candidate_vecs[rest] @ candidate_vecs[best],
            )
            _ = np.maximum(
                max_sim_to_selected[rest], sims, out=max_sim_to_selected[rest]
            )

        return [
            (self._point_to_document(point=hits[i]), hits[i].score)
            for i in selected_indices
        ]

    async def adelete_session(self, session_id: str) -> models.UpdateResult:
        return await self._client.delete(
            collection_name=self.vector_store_settings.collection_name,
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
        app_logger.info("Deleting vectors for '%s' in session %s", filename, session_id)
        return await self._client.delete(
            collection_name=self.vector_store_settings.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    ),
                    models.FieldCondition(
                        key="metadata.filename",
                        match=models.MatchValue(value=filename),
                    ),
                ]
            ),
        )

    async def aclean_up_stale_vectors(self) -> models.UpdateResult:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            seconds=settings.vector_store.ttl
        )
        app_logger.info("Stale vector cleanup — TTL cutoff: %s", cutoff.isoformat())
        return await self._client.delete(
            collection_name=self.vector_store_settings.collection_name,
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
        result = await self._client.query_points_groups(
            collection_name=self.vector_store_settings.collection_name,
            group_by="metadata.filename",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=1000,
            group_size=1,
            with_payload=models.PayloadSelectorInclude(
                include=["metadata.uploaded_at"]
            ),
            with_vectors=False,
        )

        documents: list[dict[str, str]] = []
        for group in result.groups:
            filename = group.id
            if not group.hits or not isinstance(filename, str):
                continue
            meta = cast(
                StrictMetadata, (group.hits[0].payload or {}).get("metadata", {})
            )
            uploaded_at = meta.get("uploaded_at")
            if isinstance(uploaded_at, str):
                documents.append({"filename": filename, "uploaded_at": uploaded_at})

        return documents

    async def aclose(self) -> None:
        await self._client.close()

    @staticmethod
    def _point_to_document(point: models.ScoredPoint | models.Record) -> Document:
        payload = point.payload or {}
        return Document(
            page_content=payload.get("page_content", ""),  # pyright: ignore[reportAny]
            metadata=payload.get("metadata", {}),
        )
