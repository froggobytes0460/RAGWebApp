# pyright: reportExplicitAny=none

from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, ClassVar, Self, cast

import groq
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.core.config import settings
from backend.core.llms.llm_schema import HypeQuestions
from backend.core.llms.prompt import HYPE_PROMPT, RAG_PROMPT


class LLMGroqClient(BaseModel):
    """Executes context-aware RAG logic and real-time inference streaming via Groq."""

    retryable_groq_errors: ClassVar[tuple[type[groq.APIError], ...]] = (
        groq.RateLimitError,
        groq.APIConnectionError,
        groq.APITimeoutError,
        groq.InternalServerError,
    )

    groq_client: Annotated[
        ChatGroq, Field(description="Pre-configured LangChain Groq chat client engine.")
    ]

    @property
    def runnable_lcel(
        self,
    ) -> RunnableSerializable[dict[str, Any], Any]:
        return RAG_PROMPT | self.groq_client

    @classmethod
    def from_settings(cls) -> Self:
        return cls(
            groq_client=ChatGroq(
                model=settings.llm.model_name,
                api_key=settings.llm.api_key,
                max_retries=0,  # Disabled here so tenacity handles the outer wrapper explicitly
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_output_token,
            )
        )

    async def generate_hype_questions(self, chunk: str, n: int) -> list[str]:
        chain = cast(
            RunnableSerializable[dict[str, str | int], HypeQuestions],
            HYPE_PROMPT
            | self.groq_client.with_structured_output(  # pyright: ignore[reportUnknownMemberType]
                schema=HypeQuestions
            ),
        )
        try:
            result = await chain.ainvoke(input={"chunk": chunk, "n": n})
            return result.questions[:n]
        except Exception:
            return []

    async def astream_response(
        self,
        documents: list[Document],
        question: str,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> AsyncIterator[str]:
        context = "\n\n".join(doc.page_content for doc in documents)
        payload: dict[str, str | Sequence[BaseMessage]] = {
            "context": context,
            "question": question,
            "chat_history": chat_history or [],
        }

        retrier = AsyncRetrying(
            stop=stop_after_attempt(settings.llm.max_retries),
            wait=wait_exponential_jitter(initial=2, max=10),
            retry=retry_if_exception_type(exception_types=self.retryable_groq_errors),
            reraise=True,
        )

        stream: AsyncIterator[AIMessageChunk] | None = None
        first_chunk: AIMessageChunk | None = None

        async for attempt in retrier:
            with attempt:
                stream = self.runnable_lcel.astream(payload)
                first_chunk = await stream.__anext__()

        if first_chunk and first_chunk.content:
            yield str(first_chunk.content)

        if not stream:
            return

        try:
            async for chunk in stream:
                if chunk.content:
                    yield str(chunk.content)
        except self.retryable_groq_errors as e:
            yield f"\n\n[Stream interrupted due to temporary connectivity issue: {e}]"
