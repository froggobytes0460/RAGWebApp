import threading
import time
from unittest import mock

from langchain_core.documents import Document
import pytest

from backend.core.chunking import (
    TextChunker,
    _get_cached_tokenizer,  # pyright: ignore[reportPrivateUsage]
)
import backend.core.chunking as chunking


@pytest.fixture(autouse=True)
def reset_tokenizer_cache(monkeypatch: pytest.MonkeyPatch):
    """Clear the lru_cache and reset class variables before each test."""
    _get_cached_tokenizer.cache_clear()
    TextChunker._recursive_text_splitter = None  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(chunking, "_INIT_LOCK", threading.Lock())
    yield


def test_tokenizer_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure that the tokenizer is created only once and then cached."""
    mock_from_pretrained = mock.Mock(return_value=mock.Mock())
    monkeypatch.setattr(
        "backend.core.chunking.AutoTokenizer.from_pretrained", mock_from_pretrained
    )

    t1 = _get_cached_tokenizer()
    assert mock_from_pretrained.call_count == 1

    t2 = _get_cached_tokenizer()
    assert mock_from_pretrained.call_count == 1
    assert t1 is t2


async def test_achunk_text_uses_splitter(monkeypatch: pytest.MonkeyPatch):
    """Verify that achunk_text delegates to the underlying text splitter."""
    dummy_documents = [Document(page_content="doc1"), Document(page_content="doc2")]
    expected_chunks = [Document(page_content="chunk1"), Document(page_content="chunk2")]

    class DummySplitter:
        def split_documents(self, docs: list[Document]) -> list[Document]:
            assert docs == dummy_documents
            return expected_chunks

    monkeypatch.setattr(
        TextChunker,
        "_recursive_text_splitter",
        DummySplitter(),
    )

    chunker = TextChunker()
    result = await chunker.achunk_text(dummy_documents)
    assert result == expected_chunks


def test_splitter_initialization_is_threadsafe(monkeypatch: pytest.MonkeyPatch):
    """Confirm that the splitter is created only once under heavy thread contention."""
    dummy_splitter_instance = mock.Mock()

    def delayed_ctor(
        *args,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportUnusedParameter]
        **kwargs,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportUnusedParameter]
    ) -> mock.Mock:
        time.sleep(0.05)
        return dummy_splitter_instance

    mock_splitter_ctor = mock.Mock(
        side_effect=delayed_ctor  # pyright: ignore[reportUnknownArgumentType]
    )
    monkeypatch.setattr(
        "backend.core.chunking.RecursiveCharacterTextSplitter.from_huggingface_tokenizer",
        mock_splitter_ctor,
    )

    # Pre-mock the tokenizer to isolate the splitter locking test logic
    monkeypatch.setattr(
        "backend.core.chunking._get_cached_tokenizer",
        mock.Mock(),
    )

    def load_splitter():
        return (
            TextChunker._get_splitter_recursive()  # pyright: ignore[reportPrivateUsage]
        )

    # Spawn threads simultaneously
    threads = [threading.Thread(target=load_splitter) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mock_splitter_ctor.call_count == 1
    assert (
        TextChunker._recursive_text_splitter  # pyright: ignore[reportPrivateUsage]
        is dummy_splitter_instance
    )
