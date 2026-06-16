from collections.abc import AsyncIterator
from typing import Callable, ClassVar, Protocol, Self, runtime_checkable

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from backend.core.config import settings
from backend.core.llms.groq import LLMGroqClient
from backend.core.llms.openrouter import LLMOpenRouterClient

__all__ = ["LLMClientFactory"]


@runtime_checkable
class LLMClientProto(Protocol):
    @classmethod
    def from_settings(cls) -> Self: ...

    def astream_response(
        self,
        documents: list[Document],
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> AsyncIterator[str]: ...


class LLMClientFactory:
    _registry: ClassVar[dict[str, Callable[[], LLMClientProto]]] = {
        "groq": LLMGroqClient.from_settings,
        "openrouter": LLMOpenRouterClient.from_settings,
    }

    @classmethod
    def from_settings(cls) -> LLMClientProto:
        provider = settings.llm.provider
        if provider not in cls._registry:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        client = cls._registry[provider]()
        if not isinstance(
            client, LLMClientProto
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"Provider '{provider}' (type: {type(client).__name__}) does not fully implement the required LLMClientProto interface."
            )

        return client
