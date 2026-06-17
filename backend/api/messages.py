from collections.abc import AsyncGenerator
import json
from typing import cast

from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi_utils.cbv import cbv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.api.documents import get_vector_store
from backend.api.schemas import (
    MessageHistoryItem,
    MessageRequest,
    RetrievedChunk,
    StreamChunk,
)
from backend.core.database import get_session, get_session_factory
from backend.core.ingest import StrictMetadata
from backend.core.llms import LLMClientFactory, LLMClientProto
from backend.core.models import ChatMessage, ChatSession
from backend.core.vector_store import VectorStore

messages_router = APIRouter(
    prefix="/v1/chats/{session_id}/messages",
    tags=["Messages"],
    redirect_slashes=False,
)


def _get_llm_client() -> LLMClientProto:
    return LLMClientFactory.from_settings()


@cbv(router=messages_router)
class MessageView:
    db: AsyncSession = cast(AsyncSession, Depends(dependency=get_session))
    vector_store: VectorStore = cast(VectorStore, Depends(dependency=get_vector_store))

    @messages_router.post(path="/", status_code=status.HTTP_201_CREATED)
    async def create_message(
        self,
        session_id: str,
        body: MessageRequest,
    ) -> StreamingResponse:
        """Save user message, retrieve vector context, stream LLM reply as SSE, save assistant reply."""

        retriever = self.vector_store.get_retriever(
            session_id=session_id,
            k=body.top_k,
        )
        retrieved_docs: list[Document] = await retriever.ainvoke(input=body.question)

        if not retrieved_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for this session. Upload documents first.",
            )

        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                score=float(cast(dict[str, float], doc.metadata).get("score", 0.0)),
                filename=str(
                    cast(StrictMetadata, doc.metadata).get("filename", "unknown")
                ),
                page_number=cast(dict[str, int], doc.metadata).get("page_number"),
            )
            for doc in retrieved_docs
        ]

        existing_session = await self.db.get(entity=ChatSession, ident=session_id)
        if not existing_session:
            self.db.add(instance=ChatSession(id=session_id))
            await self.db.flush()

        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=body.question,
        )
        self.db.add(instance=user_msg)
        await self.db.flush()

        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)  # pyright: ignore[reportArgumentType]
        )
        result = await self.db.exec(stmt)
        prior_messages = result.all()

        chat_history = [
            (
                HumanMessage(content=m.content)
                if m.role == "user"
                else AIMessage(content=m.content)
            )
            for m in prior_messages
            if m.id != user_msg.id
        ]

        await self.db.commit()

        llm_client = _get_llm_client()
        async_session_factory = get_session_factory()

        async def event_stream() -> AsyncGenerator[str, None]:
            answer_parts: list[str] = []
            try:
                async for text in llm_client.astream_response(
                    documents=retrieved_docs,
                    question=body.question,
                    chat_history=chat_history,
                ):
                    answer_parts.append(text)
                    payload = StreamChunk(text=text).model_dump_json()
                    yield f"event: chunk\ndata: {payload}\n\n"

                answer = "".join(answer_parts)

                async with async_session_factory() as background_db:
                    assistant_msg = ChatMessage(
                        session_id=session_id,
                        role="ai",
                        content=answer,
                        retrieved_chunks=[c.model_dump() for c in chunks],
                    )
                    background_db.add(instance=assistant_msg)
                    await background_db.commit()

                done_payload = StreamChunk(retrieved_chunks=chunks).model_dump_json()
                yield f"event: done\ndata: {done_payload}\n\n"

            except Exception as exc:
                error_payload = json.dumps(obj={"detail": str(exc)})
                yield f"event: error\ndata: {error_payload}\n\n"

        return StreamingResponse(
            content=event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @messages_router.get(path="/")
    async def list_messages(self, session_id: str) -> list[MessageHistoryItem]:
        """Retrieve the full ordered message history for a session."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)  # pyright: ignore[reportArgumentType]
        )
        result = await self.db.exec(stmt)
        messages = result.all()

        if not messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No messages found for session '{session_id}'.",
            )

        return [
            MessageHistoryItem(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ]
