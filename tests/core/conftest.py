# pyright: reportPrivateUsage=none

from collections.abc import AsyncGenerator, Callable
from pathlib import Path
import threading
from typing import override
from unittest.mock import AsyncMock, MagicMock

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore
import pytest
from pytest_mock import MockerFixture
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import UpdateResult, UpdateStatus
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core import chunking
from backend.core.chunking import TextChunker, _get_cached_tokenizer
from backend.core.config import IngestSettings, settings
from backend.core.vector_store import VectorStore


class MockEmbeddings(Embeddings):
    """Mock embeddings to prevent actual initialization of `HuggingFaceEmbeddings`."""

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * settings.vector_store.vector_size for _ in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * settings.vector_store.vector_size


@pytest.fixture(scope="session")
def fast_ingest_config() -> IngestSettings:
    return IngestSettings(
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=False,
        generate_picture_images=False,
        do_picture_classification=False,
        do_picture_description=False,
    )


@pytest.fixture(autouse=True)
def reset_tokenizer_cache(monkeypatch: pytest.MonkeyPatch):
    """Clear the lru_cache and reset class variables before each test."""
    _get_cached_tokenizer.cache_clear()
    TextChunker._recursive_text_splitter = None

    monkeypatch.setattr(chunking, "_INIT_LOCK", threading.Lock())
    yield


@pytest.fixture
def mock_qdrant_clients(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> tuple[MagicMock, AsyncMock]:
    mock_sync: MagicMock = mocker.MagicMock(spec=QdrantClient)
    mock_async: AsyncMock = mocker.AsyncMock(spec=AsyncQdrantClient)

    mock_sync.collection_exists = mocker.MagicMock(return_value=False)
    mock_async.collection_exists = mocker.AsyncMock(return_value=False)

    mock_sync.create_collection = mocker.MagicMock(return_value=True)
    mock_async.create_collection = mocker.AsyncMock(return_value=True)

    mock_sync.close = mocker.MagicMock()
    mock_async.close = mocker.AsyncMock()

    mock_success = UpdateResult(operation_id=1, status=UpdateStatus.COMPLETED)

    mock_sync.create_payload_index = mocker.MagicMock(return_value=mock_success)
    mock_async.create_payload_index = mocker.AsyncMock(return_value=mock_success)

    mock_sync.delete = mocker.MagicMock(return_value=mock_success)
    mock_async.delete = mocker.AsyncMock(return_value=mock_success)

    mock_sync_factory: Callable[..., QdrantClient] = lambda *a, **kw: mock_sync
    mock_async_factory: Callable[..., AsyncQdrantClient] = lambda *a, **kw: mock_async

    monkeypatch.setattr("backend.core.vector_store.QdrantClient", mock_sync_factory)
    monkeypatch.setattr(
        "backend.core.vector_store.AsyncQdrantClient", mock_async_factory
    )

    return mock_sync, mock_async


@pytest.fixture(params=["local_path", "remote_url"])
def vector_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    mock_qdrant_clients: tuple[MagicMock, AsyncMock],
) -> VectorStore:
    target_path_or_url: Path | str = (
        tmp_path
        if request.param == "local_path"  # pyright: ignore[reportAny]
        else "http://localhost:6333"
    )

    monkeypatch.setattr(
        settings.vector_store, "url_or_path", target_path_or_url, raising=False
    )
    monkeypatch.setattr(
        settings.vector_store, "collection_name", "test_collection", raising=False
    )
    monkeypatch.setattr(settings.vector_store, "vector_size", 1536, raising=False)
    monkeypatch.setattr(settings.vector_store, "ttl", 1, raising=False)
    monkeypatch.setattr(settings.search, "search_type", "similarity", raising=False)

    monkeypatch.setattr(
        "backend.core.vector_store._get_huggingface_embeddings",
        lambda: MockEmbeddings(),
    )

    mock_sync, mock_async = mock_qdrant_clients
    vs = VectorStore(
        client=mock_sync,
        async_client=mock_async if request.param == "remote_url" else None,
    )

    mock_lc_store = mocker.MagicMock(spec=QdrantVectorStore)
    mock_lc_store.aadd_documents = mocker.AsyncMock(return_value=["mocked_id"])

    mock_retriever = mocker.AsyncMock(spec=VectorStoreRetriever)
    mock_retriever.ainvoke.return_value = []  # pyright: ignore[reportAny]
    mock_lc_store.as_retriever.return_value = (  # pyright: ignore[reportAny]
        mock_retriever
    )

    vs._vector_store = mock_lc_store

    return vs


@pytest.fixture
async def db_engine() -> AsyncGenerator[AsyncEngine]:
    engine = create_async_engine(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(fn=SQLModel.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(fn=SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    factory = async_sessionmaker(
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session
