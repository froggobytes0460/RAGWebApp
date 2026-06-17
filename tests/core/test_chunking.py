# pyright: reportPrivateUsage=none

from collections.abc import Callable
import threading
import time
from typing import cast
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from pytest_mock import MockerFixture
from transformers import PreTrainedTokenizerBase

from backend.core.chunking import TextChunker, _get_cached_tokenizer


class TestTokenizerCache:
    def test_is_cached(self, mocker: MockerFixture) -> None:
        mock_from_pretrained: MagicMock = mocker.patch(
            "backend.core.chunking.AutoTokenizer.from_pretrained"
        )

        t1: PreTrainedTokenizerBase = _get_cached_tokenizer()
        assert mock_from_pretrained.call_count == 1

        t2: PreTrainedTokenizerBase = _get_cached_tokenizer()
        assert mock_from_pretrained.call_count == 1
        assert t1 is t2


class TestTextChunker:
    async def test_achunk_text_uses_splitter(self, mocker: MockerFixture) -> None:
        dummy_documents = [Document(page_content="doc1"), Document(page_content="doc2")]
        expected_chunks = [
            Document(page_content="chunk1"),
            Document(page_content="chunk2"),
        ]

        mock_splitter: MagicMock = mocker.MagicMock()
        mock_split_docs: MagicMock = cast(MagicMock, mock_splitter.split_documents)
        mock_split_docs.return_value = expected_chunks

        _ = mocker.patch.object(
            target=TextChunker,
            attribute="_recursive_text_splitter",
            new=mock_splitter,
        )

        chunker = TextChunker()
        result: list[Document] = await chunker.achunk_text(dummy_documents)

        assert result == expected_chunks
        mock_split_docs.assert_called_once_with(dummy_documents)

    def test_splitter_initialization_is_threadsafe(self, mocker: MockerFixture) -> None:
        dummy_splitter_instance: MagicMock = mocker.MagicMock(spec=TextSplitter)
        ctor_side_effect: Callable[..., TextSplitter] = lambda *args, **kwargs: (
            time.sleep(0.05),
            dummy_splitter_instance,
        )[1]

        mock_splitter_ctor: MagicMock = mocker.patch(
            "backend.core.chunking.RecursiveCharacterTextSplitter.from_huggingface_tokenizer",
            side_effect=ctor_side_effect,
        )

        _ = mocker.patch("backend.core.chunking._get_cached_tokenizer")

        threads: list[threading.Thread] = [
            threading.Thread(target=TextChunker._get_splitter_recursive)
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mock_splitter_ctor.call_count == 1
        assert TextChunker._recursive_text_splitter is dummy_splitter_instance
