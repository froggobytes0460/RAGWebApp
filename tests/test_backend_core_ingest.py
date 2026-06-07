from pathlib import Path
from typing import cast

import pytest
from backend.core.config import IngestSettings
from backend.core.ingest import DocumentIngestor, StrictMetadata
from langchain_core.documents import Document

FIXTURE_NEEDLES: dict[str, list[str]] = {
    "pdf": ["Christoph Auer", "Michele Dolfi", "Data selection and preparation"],
    "docx": ["Sample Document with Images", "text wrapping modes", "image handling"],
    "md": ["MD-TEST-2026", "Yasir Atiq", "PL-HYBRID-03", "o200k_base"],
    "xlsx": ["Name", "Category", "Status"],
}


@pytest.fixture(scope="session")
def fast_ingest_config() -> IngestSettings:
    return IngestSettings(
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=False,
        generate_picture_images=False,
        do_picture_classification=False,
        do_picture_description=False,
    )


async def test_ingest_unsupported_extension(tmp_path: Path) -> None:
    invalid_file = tmp_path / "sample.txt"
    _ = invalid_file.write_text("data")

    with pytest.raises(ValueError) as exc:
        _ = DocumentIngestor(file_path=invalid_file)

    assert "Unsupported file extension" in str(exc.value)


@pytest.mark.parametrize(
    argnames="fixture_key, fixture_name_attr",
    argvalues=[
        ("pdf", "fixture_pdf"),
        ("docx", "fixture_docx"),
        ("md", "fixture_md"),
        ("xlsx", "fixture_xlsx"),
    ],
)
async def test_ingest_fixtures_metadata_and_structural_integrity(
    request: pytest.FixtureRequest,
    fixture_key: str,
    fixture_name_attr: str,
    fast_ingest_config: IngestSettings,
) -> None:
    fixture_path = cast(Path, request.getfixturevalue(argname=fixture_name_attr))

    ingestor: DocumentIngestor = DocumentIngestor(
        file_path=fixture_path, config=fast_ingest_config
    )
    docs = await ingestor.ingest_async()

    assert isinstance(docs, list) and len(docs) > 0

    for doc in docs:
        assert isinstance(doc, Document)
        metadata = cast(StrictMetadata, doc.metadata)
        assert metadata.get("filename") == fixture_path.name
        assert isinstance(metadata.get("page_number"), int)

    full_text = " ".join(d.page_content for d in docs)
    for needle in FIXTURE_NEEDLES[fixture_key]:
        assert (
            needle in full_text
        ), f"Needle matching pattern '{needle}' was dropped during {fixture_key} chunk processing."


@pytest.mark.parametrize(
    argnames="metadata_dict, fallback_idx, expected_page",
    argvalues=[
        ({"page": 5}, 2, 5),
        ({}, 2, 2),
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
