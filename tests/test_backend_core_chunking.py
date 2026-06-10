import threading
import time
from typing import Any
from unittest.mock import MagicMock

from langchain_core.documents import Document
from pytest_mock import MockerFixture

from backend.core.chunking import TextChunker, _get_cached_tokenizer


def test_tokenizer_is_cached(mocker: MockerFixture) -> None:
    mock_from_pretrained: MagicMock = mocker.patch(
        "backend.core.chunking.AutoTokenizer.from_pretrained"
    )

    t1 = _get_cached_tokenizer()  # pyright: ignore[reportPrivateUsage]
    assert mock_from_pretrained.call_count == 1

    t2 = _get_cached_tokenizer()  # pyright: ignore[reportPrivateUsage]
    assert mock_from_pretrained.call_count == 1
    assert t1 is t2


async def test_achunk_text_uses_splitter(mocker: MockerFixture) -> None:
    dummy_documents = [Document(page_content="doc1"), Document(page_content="doc2")]
    expected_chunks = [Document(page_content="chunk1"), Document(page_content="chunk2")]

    mock_splitter: MagicMock = mocker.MagicMock()
    mock_split_docs: MagicMock = mock_splitter.split_documents
    mock_split_docs.return_value = expected_chunks

    mocker.patch.object(
        target=TextChunker,
        attribute="_recursive_text_splitter",
        new=mock_splitter,
    )

    chunker = TextChunker()
    result: list[Document] = await chunker.achunk_text(dummy_documents)

    assert result == expected_chunks
    mock_split_docs.assert_called_once_with(dummy_documents)


def test_splitter_initialization_is_threadsafe(mocker: MockerFixture) -> None:
    dummy_splitter_instance: MagicMock = mocker.MagicMock()

    mock_splitter_ctor: MagicMock = mocker.patch(
        "backend.core.chunking.RecursiveCharacterTextSplitter.from_huggingface_tokenizer",
        side_effect=lambda *args, **kwargs: (time.sleep(0.05), dummy_splitter_instance)[
            1
        ],
    )

    mocker.patch("backend.core.chunking._get_cached_tokenizer")

    threads: list[threading.Thread] = [
        threading.Thread(
            target=TextChunker._get_splitter_recursive  # pyright: ignore[reportPrivateUsage]
        )
        for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mock_splitter_ctor.call_count == 1
    assert (
        TextChunker._recursive_text_splitter  # pyright: ignore[reportPrivateUsage]
        is dummy_splitter_instance
    )
