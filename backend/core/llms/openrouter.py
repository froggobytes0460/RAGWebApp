from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, ClassVar, Self

from openrouter import errors
from openrouter.errors.openroutererror import OpenRouterError
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.core.config import settings
from backend.core.llms.prompt import RAG_PROMPT


class LLMOpenRouterClient(BaseModel):
    """Executes context-aware RAG logic and real-time inference streaming via openrouter."""

    retryable_openrouter_errors: ClassVar[tuple[type[OpenRouterError], ...]] = (
        errors.TooManyRequestsResponseError,
        errors.RequestTimeoutResponseError,
        errors.BadGatewayResponseError,
        errors.ServiceUnavailableResponseError,
    )

    openrouter_client: Annotated[
        ChatOpenRouter,
        Field(description="Pre-configured LangChain OpenRouter chat client engine."),
    ]

    @property
    def runnable_lcel(
        self,
    ) -> RunnableSerializable[
        dict[str, Any], Any  # pyright: ignore[reportExplicitAny]
    ]:
        return RAG_PROMPT | self.openrouter_client

    @classmethod
    def from_settings(cls) -> Self:
        return cls(
            openrouter_client=ChatOpenRouter(
                model=settings.llm.model_name,
                api_key=settings.llm.api_key,
                max_retries=0,  # Disabled here so tenacity handles the outer wrapper explicitly
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_output_token,
            )
        )

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
            retry=retry_if_exception_type(
                exception_types=self.retryable_openrouter_errors
            ),
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
        except self.retryable_openrouter_errors as e:
            yield f"\n\n[Stream interrupted due to temporary connectivity issue: {e}]"
