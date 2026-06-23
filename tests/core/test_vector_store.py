from unittest.mock import AsyncMock

from langchain_core.documents import Document
import pytest
import pytest_mock
from qdrant_client import models
from qdrant_client.http.models import QueryResponse

from backend.core.vector_store import VectorStore


class TestHypeDocs:
    async def test_ainsert_hype_docs_upserts_one_point_per_question(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()
        chunk = Document(page_content="Some passage.", metadata={"filename": "a.pdf"})
        questions = ["What is this?", "Why is it here?", "When was it written?"]

        ids = await vector_store.ainsert_hype_docs(
            pairs=[(chunk, questions)], session_id="sess-hype"
        )

        assert len(ids) == 3
        mock_qdrant_client.upsert.assert_called()  # pyright: ignore[reportAny]

    async def test_ainsert_hype_docs_fallback_on_empty_questions(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()
        chunk = Document(
            page_content="Fallback passage.", metadata={"filename": "b.pdf"}
        )

        ids = await vector_store.ainsert_hype_docs(
            pairs=[(chunk, [])], session_id="sess-fallback"
        )

        assert len(ids) == 1

    async def test_ainsert_hype_docs_stores_chunk_text_as_page_content(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()
        chunk = Document(
            page_content="Original chunk text.", metadata={"filename": "c.pdf"}
        )
        questions = ["Question about it?"]

        _ = await vector_store.ainsert_hype_docs(
            pairs=[(chunk, questions)], session_id="sess-payload"
        )

        call_args = (
            mock_qdrant_client.upsert.call_args_list
        )  # pyright: ignore[reportAny]
        assert call_args
        points: list[models.PointStruct] = call_args[0].kwargs["points"]
        payload = points[0].payload
        assert payload is not None
        assert (
            payload["page_content"] == "Original chunk text."
        )  # pyright: ignore[reportAny]
        assert "chunk_id" in payload  # pyright: ignore[reportAny]

    async def test_asearch_deduplicates_by_chunk_id(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
        mocker: pytest_mock.MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("backend.core.vector_store.settings.rerank.enabled", False)
        _ = await vector_store.ainit_collection()

        shared_chunk_id = "shared-chunk-123"

        def _make_hit(score: float) -> models.ScoredPoint:
            hit = mocker.MagicMock(spec=models.ScoredPoint)
            hit.score = score
            hit.payload = {
                "page_content": "chunk text",
                "metadata": {"chunk_id": shared_chunk_id},
                "chunk_id": shared_chunk_id,
            }
            return hit

        mock_qdrant_client.query_points = mocker.AsyncMock(
            return_value=QueryResponse(points=[_make_hit(0.9), _make_hit(0.7)])
        )

        results = await vector_store.asearch_with_scores(
            query="test", session_id="sess-dedup", k=5
        )

        assert len(results) == 1
        assert results[0][1] == 0.9


class TestVectorStoreCollection:
    async def test_initialisation(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        result = await vector_store.ainit_collection()

        assert "session_id" in result
        assert "uploaded_at" in result
        mock_qdrant_client.create_collection.assert_called_once()  # pyright: ignore[reportAny]
        assert isinstance(result["session_id"], models.UpdateResult)
        assert isinstance(result["uploaded_at"], models.UpdateResult)

    async def test_close(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()
        await vector_store.aclose()

        mock_qdrant_client.close.assert_called_once()  # pyright: ignore[reportAny]


class TestVectorStoreDocuments:
    async def test_insert_and_retrieve(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        _ = await vector_store.ainit_collection()

        doc = Document(
            page_content="Test content for vector store",
            metadata={"source": "unit-test"},
        )
        session_id = "test-session-123"

        ids = await vector_store.ainsert_docs(documents=[doc], session_id=session_id)
        assert len(ids) == 1

        mock_hit = mocker.MagicMock(spec=models.ScoredPoint)
        mock_hit.score = 0.9
        mock_hit.payload = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        mock_qdrant_client.query_points = mocker.AsyncMock(
            return_value=QueryResponse(points=[mock_hit])
        )

        results = await vector_store.asearch_with_scores(
            query="", session_id=session_id, k=1
        )
        assert len(results) == 1
        assert results[0][0].page_content == doc.page_content

    async def test_list_documents(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        _ = await vector_store.ainit_collection()

        session_id = "test-session-list"
        mock_timestamp = "2026-06-15T13:34:00Z"

        def _make_group(filename: str) -> models.PointGroup:
            hit = mocker.MagicMock(spec=models.ScoredPoint)
            hit.payload = {"metadata": {"uploaded_at": mock_timestamp}}
            return models.PointGroup(id=filename, hits=[hit])

        mock_qdrant_client.query_points_groups = mocker.AsyncMock(
            return_value=models.GroupsResult(
                groups=[
                    _make_group(filename="report.pdf"),
                    _make_group(filename="notes.txt"),
                ]
            )
        )

        documents = await vector_store.alist_documents(session_id=session_id)

        mock_qdrant_client.query_points_groups.assert_called_once()  # pyright: ignore[reportAny]
        assert len(documents) == 2

        filenames = [doc["filename"] for doc in documents]
        assert "report.pdf" in filenames
        assert "notes.txt" in filenames
        assert documents[0]["uploaded_at"] == mock_timestamp

    async def test_delete_document(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()

        session_id = "test-session-delete"
        filename = "stale_data.pdf"

        result = await vector_store.adelete_document(
            session_id=session_id, filename=filename
        )

        mock_qdrant_client.delete.assert_called_once()  # pyright: ignore[reportAny]
        assert isinstance(result, models.UpdateResult)

    async def test_delete_session(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()

        session_id = "delete-me"
        _ = await vector_store.adelete_session(session_id=session_id)

        mock_qdrant_client.delete.assert_called_once()  # pyright: ignore[reportAny]

    async def test_cleanup_stale_vectors(
        self,
        vector_store: VectorStore,
        mock_qdrant_client: AsyncMock,
    ) -> None:
        _ = await vector_store.ainit_collection()

        doc = Document(page_content="Stale vector", metadata={"source": "unit-test"})
        _ = await vector_store.ainsert_docs(documents=[doc], session_id="stale-session")

        result = await vector_store.aclean_up_stale_vectors()

        assert result.status == models.UpdateStatus.COMPLETED
        mock_qdrant_client.delete.assert_called_once()  # pyright: ignore[reportAny]
