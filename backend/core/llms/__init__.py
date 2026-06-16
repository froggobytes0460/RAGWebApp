from typing import ClassVar

from backend.core.config import settings
from backend.core.llms.base import LLMCLientProto
from backend.core.llms.groq import LLMGroqClient

__all__ = ["LLMClientFactory"]


class LLMClientFactory:
    _registry: ClassVar[dict[str, type[LLMCLientProto]]] = {
        "groq": LLMGroqClient,
    }

    @classmethod
    def from_settings(cls) -> LLMCLientProto:
        client_cls = cls._registry[settings.llm.provider]
        return client_cls.from_settings()
