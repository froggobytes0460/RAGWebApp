import asyncio
from functools import lru_cache
import threading
from typing import ClassVar, cast

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from transformers import PreTrainedTokenizerBase
from transformers.models import AutoTokenizer

from backend.core.config import settings

_INIT_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _get_cached_tokenizer() -> PreTrainedTokenizerBase:
    return cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            pretrained_model_name_or_path=settings.vector_store.tokenizer_model
        ),
    )


class TextChunker:
    _recursive_text_splitter: ClassVar[TextSplitter | None] = None

    @classmethod
    def _get_splitter_recursive(cls) -> TextSplitter:
        if cls._recursive_text_splitter is None:
            with _INIT_LOCK:
                if cls._recursive_text_splitter is None:
                    cls._recursive_text_splitter = (
                        RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                            tokenizer=_get_cached_tokenizer()
                        )
                    )
        return cls._recursive_text_splitter

    async def achunk_text(self, documents: list[Document]) -> list[Document]:
        return await asyncio.to_thread(
            self._get_splitter_recursive().split_documents, documents
        )
