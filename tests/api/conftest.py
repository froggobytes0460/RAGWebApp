# pyright: reportExplicitAny=none
# pyright: reportAny=none

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient
from langchain_core.documents import Document
import pytest
import pytest_mock
from qdrant_client import models
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.api import app
from backend.api.documents import get_vector_store
from backend.api.messages import get_llm_client
from backend.api.state import AppState
from backend.core.database import get_session


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


def _make_mock_vector_store(
    docs: list[Document], mocker: pytest_mock.MockerFixture
) -> MagicMock:
    mock_vs = mocker.MagicMock()
    scored_docs = [(doc, 0.95) for doc in docs]
    mock_vs.asearch_with_scores = mocker.AsyncMock(return_value=scored_docs)
    return mock_vs


def _make_mock_llm(answer: str, mocker: pytest_mock.MockerFixture) -> MagicMock:
    async def _stream(
        *_: Any, **__: Any  # pyright: ignore[reportUnusedParameter]
    ) -> AsyncIterator[str]:
        yield answer

    mock_llm = mocker.MagicMock()
    mock_llm.astream_response = _stream
    mock_llm.generate_hype_questions = mocker.AsyncMock(
        return_value=["What is X?", "How does Y work?", "When did Z occur?"]
    )
    return mock_llm


def _make_mock_docs() -> list[Document]:
    return [
        Document(
            page_content="relevant context",
            metadata={"filename": "doc.pdf", "page_number": 1, "score": 0.95},
        )
    ]


@pytest.fixture
async def client(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    mocker: pytest_mock.MockerFixture,
) -> AsyncGenerator[AsyncClient]:
    mock_vs = _make_mock_vector_store(docs=_make_mock_docs(), mocker=mocker)
    mock_llm = _make_mock_llm(answer="default answer", mocker=mocker)

    background_factory = async_sessionmaker(
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )
    _ = mocker.patch(
        "backend.api.messages.get_session_factory",
        return_value=background_factory,
    )

    async def _override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    app.dependency_overrides[get_vector_store] = lambda: mock_vs
    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    app.typed_state = AppState()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        yield http_client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_result() -> models.UpdateResult:
    return models.UpdateResult(operation_id=1, status=models.UpdateStatus.COMPLETED)


@pytest.fixture
def mock_pdf_bytes() -> bytes:
    return b"%PDF-1.4 fake pdf content"
