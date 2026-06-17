# pyright: reportAny=false

from sqlmodel.ext.asyncio.session import AsyncSession

import backend.core.models  # pyright: ignore[reportUnusedImport]
from backend.core.models import ChatMessage, ChatSession


class TestChatSession:
    async def test_create_session_with_explicit_id(
        self, db_session: AsyncSession
    ) -> None:
        session = ChatSession(id="sess-abc")
        db_session.add(instance=session)
        await db_session.commit()

        result = await db_session.get(entity=ChatSession, ident="sess-abc")
        assert result is not None
        assert result.id == "sess-abc"
        assert result.created_at is not None

    async def test_create_session_auto_id(self, db_session: AsyncSession) -> None:
        session = ChatSession()
        db_session.add(session)
        await db_session.commit()

        assert session.id is not None
        assert len(session.id) == 36


class TestChatMessage:
    async def _create_session(self, db_session: AsyncSession, session_id: str) -> None:
        db_session.add(instance=ChatSession(id=session_id))
        await db_session.flush()

    async def test_create_user_message(self, db_session: AsyncSession) -> None:
        await self._create_session(db_session, session_id="sess-1")
        msg = ChatMessage(session_id="sess-1", role="user", content="hello")
        db_session.add(instance=msg)
        await db_session.commit()

        assert msg.id is not None
        assert msg.retrieved_chunks is None

    async def test_create_assistant_message_with_chunks(
        self, db_session: AsyncSession
    ) -> None:
        await self._create_session(db_session, session_id="sess-2")
        chunks = [
            {
                "content": "ctx text",
                "score": 0.9,
                "filename": "doc.pdf",
                "page_number": 1,
            }
        ]
        msg = ChatMessage(
            session_id="sess-2",
            role="ai",
            content="the answer",
            retrieved_chunks=chunks,
        )
        db_session.add(instance=msg)
        await db_session.commit()

        refreshed = await db_session.get(entity=ChatMessage, ident=msg.id)
        assert refreshed is not None
        assert refreshed.retrieved_chunks == chunks

    async def test_multiple_messages_ordered_by_creation(
        self, db_session: AsyncSession
    ) -> None:
        await self._create_session(db_session, session_id="sess-3")
        for content in ("first", "second", "third"):
            db_session.add(
                ChatMessage(session_id="sess-3", role="user", content=content)
            )
        await db_session.commit()

        from sqlmodel import select

        result = await db_session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == "sess-3")
            .order_by(ChatMessage.created_at)  # pyright: ignore[reportArgumentType]
        )
        msgs = result.all()
        assert [m.content for m in msgs] == ["first", "second", "third"]
