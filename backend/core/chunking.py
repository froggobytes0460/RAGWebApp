import asyncio
from functools import lru_cache

from langchain_core.documents import Document
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

from backend.core.config import settings
from backend.core.logging import app_logger


@lru_cache(maxsize=1)
def _get_cached_tokenizer() -> Tokenizer:
    app_logger.info("Loading tokenizer: %s", settings.text_chunk.tokenizer_model)
    return Tokenizer.from_pretrained(settings.text_chunk.tokenizer_model)


@lru_cache(maxsize=1)
def _get_splitter() -> TextSplitter:
    tokenizer = _get_cached_tokenizer()
    return TextSplitter.from_huggingface_tokenizer(  # pyright: ignore[reportUnknownMemberType]
        tokenizer,
        capacity=settings.text_chunk.chunk_size,
        overlap=settings.text_chunk.chunk_overlap,
    )


class TextChunker:
    @staticmethod
    def _chunk_document(doc: Document) -> list[Document]:
        splitter = _get_splitter()
        return [
            Document(page_content=p, metadata=doc.metadata)
            for p in splitter.chunks(text=doc.page_content)
            if p.strip()
        ]

    @classmethod
    def _split_documents(cls, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            result.extend(cls._chunk_document(doc))
        app_logger.debug(
            "Split %d document(s) into %d chunk(s)", len(documents), len(result)
        )
        return result

    async def achunk_text(self, documents: list[Document]) -> list[Document]:
        return await asyncio.to_thread(self._split_documents, documents)
