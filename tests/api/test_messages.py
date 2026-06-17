# pyright: reportAny=none
# pyright: reportPrivateUsage=none

import json
from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient
import pytest_mock
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import backend.core.models  # pyright: ignore[reportUnusedImport]
from backend.api.app import app
from backend.api.documents import get_vector_store
from backend.core.database import get_session
from backend.core.models import ChatMessage, ChatSession

from tests.api.conftest import _make_mock_llm, _make_mock_vector_store


def _parse_sse(text: str) -> list[dict[str, str]]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            current["data"] = line.removeprefix("data:").strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


class TestCreateMessage:
    async def test_returns_201_with_answer(
        self,
        client: AsyncClient,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_llm = _make_mock_llm(answer="The answer is 42.", mocker=mocker)
        _ = mocker.patch("backend.api.messages._get_llm_client", return_value=mock_llm)

        resp = await client.post(
            url="/api/v1/chats/sess-test/messages/",
            json={"question": "What is the answer?"},
        )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        chunk_events = [e for e in events if e.get("event") == "chunk"]
        done_events = [e for e in events if e.get("event") == "done"]

        assert len(chunk_events) >= 1
        full_answer = "".join(json.loads(e["data"])["text"] for e in chunk_events)
        assert full_answer == "The answer is 42."

        assert len(done_events) == 1
        done_data = json.loads(done_events[0]["data"])
        assert len(done_data["retrieved_chunks"]) == 1
        assert done_data["retrieved_chunks"][0]["filename"] == "doc.pdf"

    async def test_404_when_no_documents_retrieved(
        self,
        db_session: AsyncSession,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_vs = _make_mock_vector_store(docs=[], mocker=mocker)

        async def _override_get_session() -> AsyncGenerator[AsyncSession]:
            yield db_session

        app.dependency_overrides[get_session] = _override_get_session
        app.dependency_overrides[get_vector_store] = lambda: mock_vs

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as http_client:
            resp = await http_client.post(
                url="/api/v1/chats/sess-empty/messages/",
                json={"question": "anything?"},
            )

        app.dependency_overrides.clear()
        assert resp.status_code == 404

    async def test_messages_persisted_to_db(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        db_engine: AsyncEngine,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_llm = _make_mock_llm(answer="persisted answer", mocker=mocker)
        _ = mocker.patch("backend.api.messages._get_llm_client", return_value=mock_llm)

        background_factory = async_sessionmaker(
            bind=db_engine,  # type: ignore[arg-type]
            class_=AsyncSession,
            expire_on_commit=False,
        )
        _ = mocker.patch(
            "backend.api.messages.get_session_factory",
            return_value=background_factory,
        )

        resp = await client.post(
            url="/api/v1/chats/sess-persist/messages/",
            json={"question": "save me?"},
        )
        assert resp.status_code == 200

        result = await db_session.exec(
            select(ChatMessage).where(ChatMessage.session_id == "sess-persist")
        )
        msgs = result.all()
        assert len(msgs) == 2
        roles = {m.role for m in msgs}
        assert roles == {"user", "ai"}

    async def test_session_row_created(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        db_engine: AsyncEngine,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_llm = _make_mock_llm(answer="ok", mocker=mocker)
        _ = mocker.patch("backend.api.messages._get_llm_client", return_value=mock_llm)

        background_factory = async_sessionmaker(
            bind=db_engine,  # type: ignore[arg-type]
            class_=AsyncSession,
            expire_on_commit=False,
        )
        _ = mocker.patch(
            "backend.api.messages.get_session_factory",
            return_value=background_factory,
        )

        resp = await client.post(
            url="/api/v1/chats/new-session/messages/",
            json={"question": "hello?"},
        )
        assert resp.status_code == 200

        session_row = await db_session.get(entity=ChatSession, ident="new-session")
        assert session_row is not None


class TestListMessages:
    async def test_returns_ordered_history(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        db_session.add(instance=ChatSession(id="sess-history"))
        await db_session.flush()
        db_session.add(
            instance=ChatMessage(session_id="sess-history", role="user", content="q1")
        )
        db_session.add(
            instance=ChatMessage(session_id="sess-history", role="ai", content="a1")
        )
        await db_session.commit()

        resp = await client.get(url="/api/v1/chats/sess-history/messages/")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["role"] == "user"
        assert body[0]["content"] == "q1"
        assert body[1]["role"] == "ai"

    async def test_404_for_unknown_session(self, client: AsyncClient) -> None:
        resp = await client.get(url="/api/v1/chats/nonexistent-session/messages/")
        assert resp.status_code == 404
