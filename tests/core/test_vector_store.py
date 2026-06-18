from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document
import pytest_mock
from qdrant_client import models

from backend.core.vector_store import VectorStore


def _get_active_mock(
    vector_store: VectorStore, mock_qdrant_clients: tuple[MagicMock, AsyncMock]
) -> MagicMock | AsyncMock:
    mock_sync, mock_async = mock_qdrant_clients
    return mock_async if vector_store.async_client is not None else mock_sync


class TestVectorStoreCollection:
    async def test_initialisation(
        self,
        vector_store: VectorStore,
        mock_qdrant_clients: tuple[MagicMock, AsyncMock],
    ) -> None:
        active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
        result = await vector_store.ainit_collection()

        assert "session_id" in result
        assert "uploaded_at" in result
        active_mock.create_collection.assert_called_once()  # pyright: ignore[reportAny]
        assert isinstance(result["session_id"], models.UpdateResult)
        assert isinstance(result["uploaded_at"], models.UpdateResult)

    async def test_close(
        self,
        vector_store: VectorStore,
        mock_qdrant_clients: tuple[MagicMock, AsyncMock],
    ) -> None:
        active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
        _ = await vector_store.ainit_collection()
        await vector_store.aclose()

        active_mock.close.assert_called_once()  # pyright: ignore[reportAny]


class TestVectorStoreDocuments:
    async def test_insert_and_retrieve(
        self, vector_store: VectorStore, mocker: pytest_mock.MockerFixture
    ) -> None:
        _ = await vector_store.ainit_collection()

        doc = Document(
            page_content="Test content for vector store",
            metadata={"source": "unit-test"},
        )
        session_id = "test-session-123"

        ids = await vector_store.ainsert_docs(documents=[doc], session_id=session_id)
        assert len(ids) == 1

        _ = mocker.patch.object(
            target=vector_store.vector_store,
            attribute="asimilarity_search_with_relevance_scores",
            new=mocker.AsyncMock(return_value=[(doc, 0.9)]),
        )

        results = await vector_store.asearch_with_scores(
            query="", session_id=session_id, k=1
        )
        assert len(results) == 1
        assert results[0][0].page_content == doc.page_content

    async def test_list_documents(
        self,
        vector_store: VectorStore,
        mock_qdrant_clients: tuple[MagicMock, AsyncMock],
    ) -> None:
        active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
        _ = await vector_store.ainit_collection()

        session_id = "test-session-list"
        mock_timestamp = "2026-06-15T13:34:00Z"

        mock_records = [
            models.Record(
                id=1,
                payload={
                    "metadata": {
                        "filename": "report.pdf",
                        "uploaded_at": mock_timestamp,
                    }
                },
            ),
            models.Record(
                id=2,
                payload={
                    "metadata": {
                        "filename": "report.pdf",
                        "uploaded_at": mock_timestamp,
                    }
                },
            ),
            models.Record(
                id=3,
                payload={
                    "metadata": {"filename": "notes.txt", "uploaded_at": mock_timestamp}
                },
            ),
        ]

        active_mock.scroll.return_value = (  # pyright: ignore[reportAny]
            mock_records,
            None,
        )

        documents = await vector_store.alist_documents(session_id=session_id)

        active_mock.scroll.assert_called_once()  # pyright: ignore[reportAny]
        assert len(documents) == 2

        filenames = [doc["filename"] for doc in documents]
        assert "report.pdf" in filenames
        assert "notes.txt" in filenames
        assert documents[0]["uploaded_at"] == mock_timestamp

    async def test_delete_document(
        self,
        vector_store: VectorStore,
        mock_qdrant_clients: tuple[MagicMock, AsyncMock],
    ) -> None:
        active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
        _ = await vector_store.ainit_collection()

        session_id = "test-session-delete"
        filename = "stale_data.pdf"

        result = await vector_store.adelete_document(
            session_id=session_id, filename=filename
        )

        active_mock.delete.assert_called_once()  # pyright: ignore[reportAny]
        assert isinstance(result, models.UpdateResult)

    async def test_delete_session(
        self,
        vector_store: VectorStore,
        mock_qdrant_clients: tuple[MagicMock, AsyncMock],
    ) -> None:
        active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
        _ = await vector_store.ainit_collection()

        session_id = "delete-me"
        _ = await vector_store.adelete_session(session_id=session_id)

        active_mock.delete.assert_called_once()  # pyright: ignore[reportAny]

    async def test_cleanup_stale_vectors(
        self,
        vector_store: VectorStore,
        mock_qdrant_clients: tuple[MagicMock, AsyncMock],
    ) -> None:
        active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
        _ = await vector_store.ainit_collection()

        doc = Document(page_content="Stale vector", metadata={"source": "unit-test"})
        _ = await vector_store.ainsert_docs(documents=[doc], session_id="stale-session")

        result = await vector_store.aclean_up_stale_vectors()

        assert result.status == models.UpdateStatus.COMPLETED
        active_mock.delete.assert_called_once()  # pyright: ignore[reportAny]
