"""Document ingestion utilities for loading and processing files efficiently."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, ClassVar, Self, TypeAlias, cast

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ExcelFormatOption,
    FormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    WordFormatOption,
)
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from pydantic.fields import Field, computed_field
from pydantic.functional_validators import AfterValidator, model_validator
from pydantic.main import BaseModel
from pydantic.types import FilePath

from backend.core.config import IngestSettings, settings


def verify_file_integrity(path: Path) -> Path:
    """Prevent file spoofing."""
    try:
        with path.open("rb") as f:
            head: bytes = f.read(4)
    except OSError as err:
        raise ValueError(
            f"File validation failed: Unable to read file bytes. Error: {str(err)}"
        )

    suffix = path.suffix.lower()

    if suffix == ".pdf" and not head.startswith(b"%PDF"):
        raise ValueError(
            "File signature mismatch: File extension claims to be .pdf but headers do not match specification."
        )

    elif suffix in {".docx", ".xlsx"} and not head.startswith(b"PK"):
        raise ValueError(
            f"File signature mismatch: '{suffix}' container structure is corrupted or invalid."
        )

    return path


# Type Aliases
VerifiedFilePath: TypeAlias = Annotated[FilePath, AfterValidator(verify_file_integrity)]
MetadataValue: TypeAlias = (
    str | int | float | bool | None | list[object] | dict[str, object]
)
StrictMetadata: TypeAlias = dict[str, MetadataValue]


class DocumentIngestor(BaseModel):
    """Asynchronously ingests documents from various file formats with schema protection."""

    allowed_extensions: ClassVar[set[str]] = {
        ".pdf",
        ".docx",
        ".md",
        ".xlsx",
    }
    file_path: Annotated[
        VerifiedFilePath, Field(description="File path of the Document to be ingested.")
    ]
    config: IngestSettings = settings.ingest

    @computed_field
    @property
    def extension(self) -> str:
        return self.file_path.suffix.lower()

    @model_validator(mode="after")
    def file_extention_in_allowed_extentions(self) -> Self:
        if self.extension not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported file extension: {self.extension}. Supported formats: {', '.join(sorted(self.allowed_extensions))}"
            )
        return self

    def _get_optimized_loader(self) -> DoclingLoader:
        pdf_pipeline_options = PdfPipelineOptions(
            do_ocr=self.config.do_ocr,
            do_table_structure=self.config.do_table_structure,
            generate_page_images=self.config.generate_page_images,
            generate_picture_images=self.config.generate_picture_images,
        )

        other_pipeline_options = ConvertPipelineOptions(
            do_picture_classification=self.config.do_picture_classification,
            do_picture_description=self.config.do_picture_description,
        )

        format_options: dict[InputFormat, FormatOption] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=other_pipeline_options),
            InputFormat.MD: MarkdownFormatOption(
                pipeline_options=other_pipeline_options
            ),
            InputFormat.XLSX: ExcelFormatOption(
                pipeline_options=other_pipeline_options
            ),
        }

        return DoclingLoader(
            file_path=[str(self.file_path)],
            converter=DocumentConverter(format_options=format_options),
        )

    async def ingest_lazy(self) -> AsyncIterator[Document]:
        loader: DoclingLoader = self._get_optimized_loader()

        current_page_tracking = 1

        async for document in loader.alazy_load():
            metadata: StrictMetadata = cast(StrictMetadata, document.metadata)

            resolved_page = self._extract_page_number(
                metadata, fallback_index=current_page_tracking
            )
            current_page_tracking = resolved_page

            metadata["filename"] = self.file_path.name
            metadata["page_number"] = resolved_page

            yield document

    async def ingest_async(self) -> list[Document]:
        return [doc async for doc in self.ingest_lazy()]

    @staticmethod
    def _extract_page_number(
        metadata: StrictMetadata,
        fallback_index: int,
    ) -> int:
        page_value: MetadataValue = metadata.get("page") or metadata.get("page_number")
        if page_value is None:
            return fallback_index

        try:
            val = int(str(page_value))
            return val if val > 0 else fallback_index
        except (ValueError, TypeError):
            return fallback_index
