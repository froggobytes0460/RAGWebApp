from pathlib import Path
from typing import Self
from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest
from pytest_mock.plugin import MockerFixture

from backend.core.config import IngestSettings
from backend.core.ingest import DocumentIngestor, StrictMetadata


class AsyncDocIterator:
    def __init__(self, docs: list[Document]):
        self._docs: list[Document] = docs
        self._index: int = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Document:
        if self._index >= len(self._docs):
            raise StopAsyncIteration
        doc: Document = self._docs[self._index]
        self._index += 1
        return doc


async def test_ingest_unsupported_extension(tmp_path: Path) -> None:
    invalid_file = tmp_path / "sample.txt"
    _ = invalid_file.write_text("data")

    with pytest.raises(ValueError) as exc:
        _ = DocumentIngestor(file_path=invalid_file)

    assert "Unsupported file extension" in str(exc.value)


async def test_ingest_async_success(
    fast_ingest_config: IngestSettings,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    mock_docs = [
        Document(page_content="test content 1", metadata={"page": "1"}),
        Document(page_content="test content 2", metadata={"page": "2"}),
    ]

    pdf_file = tmp_path / "test.pdf"
    _ = pdf_file.write_bytes(b"%PDF-1.4\n%test")

    mock_get: MagicMock = mocker.patch.object(
        target=DocumentIngestor, attribute="_get_optimized_loader"
    )
    mock_loader = mocker.MagicMock()
    mock_alazy_load: MagicMock = mock_loader.alazy_load  # pyright: ignore[reportAny]
    mock_alazy_load.return_value = AsyncDocIterator(docs=mock_docs)

    mock_get.return_value = mock_loader

    ingest: DocumentIngestor = DocumentIngestor(
        file_path=pdf_file, config=fast_ingest_config
    )

    result: list[Document] = await ingest.ingest_async()

    assert result == mock_docs
    assert len(result) == 2
    mock_get.assert_called_once()


@pytest.mark.parametrize(
    argnames="metadata_dict,fallback_idx,expected_page",
    argvalues=[
        ({"page": "3"}, 1, 3),
        ({"page_number": 5}, 2, 5),
        ({}, 4, 4),
    ],
)
def test_page_number_extraction(
    metadata_dict: StrictMetadata, fallback_idx: int, expected_page: int
) -> None:
    assert (
        DocumentIngestor._extract_page_number(  # pyright: ignore[reportPrivateUsage]
            metadata=metadata_dict, fallback_index=fallback_idx
        )
        == expected_page
    )
