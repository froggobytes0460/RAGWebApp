from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document
from qdrant_client.models import UpdateStatus

from backend.core.vector_store import VectorStore


def _get_active_mock(
    vector_store: VectorStore, mock_qdrant_clients: tuple[MagicMock, AsyncMock]
) -> MagicMock | AsyncMock:
    mock_sync, mock_async = mock_qdrant_clients
    return mock_async if vector_store.async_client is not None else mock_sync


async def test_collection_initialisation(
    vector_store: VectorStore, mock_qdrant_clients: tuple[MagicMock, AsyncMock]
) -> None:
    active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
    result = await vector_store.ainit_collection()

    assert "session_id" in result
    assert "uploaded_at" in result
    active_mock.create_collection.assert_called_once()  # pyright: ignore[reportAny]


async def test_insert_and_retrieve(vector_store: VectorStore) -> None:
    _ = await vector_store.ainit_collection()

    doc = Document(
        page_content="Test content for vector store",
        metadata={"source": "unit-test"},
    )
    session_id = "test-session-123"

    ids = await vector_store.ainsert_docs(documents=[doc], session_id=session_id)
    assert len(ids) == 1

    retriever = vector_store.get_retriever(session_id=session_id, k=1)

    retriever.ainvoke.return_value = [  # pyright: ignore[reportAttributeAccessIssue]
        doc
    ]
    results = await retriever.ainvoke("")
    assert len(results) == 1
    assert results[0].page_content == doc.page_content


async def test_delete_session(
    vector_store: VectorStore, mock_qdrant_clients: tuple[MagicMock, AsyncMock]
) -> None:
    active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
    _ = await vector_store.ainit_collection()

    session_id = "delete-me"
    _ = await vector_store.adelete_session(session_id=session_id)

    active_mock.delete.assert_called_once()  # pyright: ignore[reportAny]


async def test_cleanup_stale_vectors(
    vector_store: VectorStore, mock_qdrant_clients: tuple[MagicMock, AsyncMock]
) -> None:
    active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
    _ = await vector_store.ainit_collection()

    doc = Document(page_content="Stale vector", metadata={"source": "unit-test"})
    _ = await vector_store.ainsert_docs(documents=[doc], session_id="stale-session")

    result = await vector_store.aclean_up_stale_vectors()

    assert result.status == UpdateStatus.COMPLETED
    active_mock.delete.assert_called_once()  # pyright: ignore[reportAny]


async def test_close(
    vector_store: VectorStore, mock_qdrant_clients: tuple[MagicMock, AsyncMock]
) -> None:
    active_mock = _get_active_mock(vector_store, mock_qdrant_clients)
    _ = await vector_store.ainit_collection()
    await vector_store.aclose()

    active_mock.close.assert_called_once()  # pyright: ignore[reportAny]
