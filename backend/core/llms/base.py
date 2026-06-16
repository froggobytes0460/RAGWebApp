from collections.abc import AsyncIterator
from typing import Protocol, Self

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class LLMCLientProto(Protocol):
    @classmethod
    def from_settings(cls) -> Self: ...

    def astream_response(
        self,
        documents: list[Document],
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> AsyncIterator[str]: ...
