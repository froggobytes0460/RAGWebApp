# pyright: reportPrivateUsage=none

import threading
from unittest.mock import MagicMock

from langchain_core.documents import Document
from pytest_mock import MockerFixture
from tokenizers import Tokenizer

from backend.core.chunking import TextChunker, _get_cached_tokenizer, _get_splitter


class TestTokenizerCache:
    def test_is_cached(self, mocker: MockerFixture) -> None:
        mock_from_pretrained: MagicMock = mocker.patch(
            "backend.core.chunking.Tokenizer.from_pretrained"
        )

        t1: Tokenizer = _get_cached_tokenizer()
        assert mock_from_pretrained.call_count == 1

        t2: Tokenizer = _get_cached_tokenizer()
        assert mock_from_pretrained.call_count == 1
        assert t1 is t2

    def test_splitter_is_cached(self, mocker: MockerFixture) -> None:
        mock_factory: MagicMock = mocker.patch(
            "backend.core.chunking.TextSplitter.from_huggingface_tokenizer"
        )
        mock_factory.return_value = MagicMock()

        s1 = _get_splitter()
        s2 = _get_splitter()
        assert mock_factory.call_count == 1
        assert s1 is s2


class TestTextChunker:
    async def test_achunk_text_returns_documents(self, mocker: MockerFixture) -> None:
        expected = [Document(page_content="chunk1"), Document(page_content="chunk2")]
        _ = mocker.patch.object(
            target=TextChunker, attribute="_split_documents", return_value=expected
        )

        result = await TextChunker().achunk_text([Document(page_content="input")])
        assert result == expected

    def test_split_documents_concurrent_calls_are_safe(self) -> None:
        docs = [Document(page_content="hello world " * 10)]
        results: list[list[Document]] = []
        lock = threading.Lock()

        def run() -> None:
            r = TextChunker._split_documents(docs)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=run) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, list) for r in results)

    async def test_achunk_text_filters_empty_chunks(
        self, mocker: MockerFixture
    ) -> None:
        mock_splitter = MagicMock()
        _ = mock_splitter.chunks.return_value = [  # pyright: ignore[reportAny]
            "  ",
            "real content",
        ]
        _ = mocker.patch(
            "backend.core.chunking._get_splitter", return_value=mock_splitter
        )

        result = await TextChunker().achunk_text([Document(page_content="anything")])
        contents = [d.page_content for d in result]
        assert "  " not in contents
        assert "real content" in contents
