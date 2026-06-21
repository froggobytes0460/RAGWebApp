# pyright: reportPrivateUsage=none

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import UpdateResult, UpdateStatus
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core.chunking import _get_cached_tokenizer, _get_splitter
from backend.core.config import settings
from backend.core.vector_store import VectorStore


@pytest.fixture(autouse=True)
def reset_tokenizer_cache():
    """Clear lru_caches before each test to prevent state bleed."""
    _get_cached_tokenizer.cache_clear()
    _get_splitter.cache_clear()
    yield
    _get_cached_tokenizer.cache_clear()
    _get_splitter.cache_clear()


@pytest.fixture
def mock_qdrant_client(mocker: MockerFixture) -> AsyncMock:
    mock_success = UpdateResult(operation_id=1, status=UpdateStatus.COMPLETED)
    mock = mocker.AsyncMock(spec=AsyncQdrantClient)
    mock.collection_exists = mocker.AsyncMock(return_value=False)
    mock.create_collection = mocker.AsyncMock(return_value=True)
    mock.close = mocker.AsyncMock()
    mock.create_payload_index = mocker.AsyncMock(return_value=mock_success)
    mock.delete = mocker.AsyncMock(return_value=mock_success)
    mock.upsert = mocker.AsyncMock(return_value=mock_success)
    return mock


@pytest.fixture
def vector_store(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    mock_qdrant_client: AsyncMock,
) -> VectorStore:
    monkeypatch.setattr(
        settings.vector_store, "collection_name", "test_collection", raising=False
    )
    monkeypatch.setattr(settings.vector_store, "vector_size", 1536, raising=False)
    monkeypatch.setattr(settings.vector_store, "ttl", 1, raising=False)
    monkeypatch.setattr(settings.search, "search_type", "similarity", raising=False)

    monkeypatch.setattr(
        "backend.core.vector_store._embed",
        lambda texts: [[0.1] * settings.vector_store.vector_size for _ in texts],
    )

    return VectorStore(mock_qdrant_client, vector_store_settings=settings.vector_store)


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
