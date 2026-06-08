from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from pydantic.fields import Field
from pydantic.functional_validators import field_validator, model_validator
from pydantic.main import BaseModel
from pydantic.types import DirectoryPath, SecretStr
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parents[2]
ENV_FILE = BASE_DIR / ".env"


class IngestSettings(BaseModel):
    max_file_size: Annotated[
        int, Field(ge=1, description="Maximum filesize for each file.")
    ] = 50
    do_ocr: Annotated[
        bool, Field(description="Enable optical character recognition for scans.")
    ] = False
    do_table_structure: Annotated[
        bool, Field(description="Extract structural table grids into data layers.")
    ] = False
    generate_page_images: Annotated[
        bool, Field(description="Render individual page images into memory.")
    ] = False
    generate_picture_images: Annotated[
        bool, Field(description="Extract standalone image crops from documents.")
    ] = False
    do_picture_classification: Annotated[
        bool, Field(description="Run classification algorithms on embedded graphics.")
    ] = False
    do_picture_description: Annotated[
        bool, Field(description="Generate textual descriptions for layout pictures.")
    ] = False

    @model_validator(mode="after")
    def validate_picture_dependencies(self) -> Self:
        if not self.generate_picture_images:
            self.do_picture_classification = False
            self.do_picture_description = False
        return self


class VectorStoreSettings(BaseModel):
    """Settings for vector store."""

    path: Annotated[
        DirectoryPath,
        Field(
            description="Local directory path where Chroma DB will persist its data files."
        ),
    ] = (
        BASE_DIR / "chroma_db"
    )

    collection_name: Annotated[
        str,
        Field(
            description="The target collection segment name inside the Chroma database instance."
        ),
    ] = "rag_documents"

    embedding_model: Annotated[
        str,
        Field(description="HuggingFace model ID used to compute text vector profiles."),
    ] = "sentence-transformers/all-MiniLM-L6-v2"

    tokenizer_model: Annotated[
        str,
        Field(description="HuggingFace model ID used to tokenize."),
    ] = "BAAI/bge-large-en-v1.5"

    chunk_size: Annotated[int, Field(gt=0, description="Size of each chunk size.")]
    chunk_overlap: Annotated[
        int, Field(ge=0, description="Amount of tokens allowed to be ")
    ]

    @field_validator("path", mode="before")
    @classmethod
    def ensure_directory_exists(cls, v: str | Path) -> Path:
        directory = Path(v)
        if (
            not directory.is_file() and not directory.exists()
        ):  # Let DirectoryPath handle filepaths.
            directory.mkdir(parents=True)
        return directory

    @model_validator(mode="after")
    def chunk_overlap_less_than_size(self) -> Self:
        if not self.chunk_overlap < self.chunk_size:
            raise PydanticCustomError(
                "value_error",
                "chunk_overlap ({overlap}) must be strictly less than chunk_size ({size}).",
                {"overlap": self.chunk_overlap, "size": self.chunk_size},
            )

        return self


class SearchSettings(BaseModel):
    """Settings for search."""

    top_k: Annotated[
        int,
        Field(ge=1, description="Number of best-matching chunks for system retrieval."),
    ]

    search_type: Annotated[
        Literal["similarity", "mmr", "similarity_score_threshold"],
        Field(
            description="The strategy algorithm LangChain employs to pull related reference context."
        ),
    ] = "similarity"

    score_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Relevance confidence minimum; only used if search_type is set to threshold matching.",
        ),
    ] = 0.5


class LLMSettings(BaseModel):
    """Settings for LLM."""

    temperature: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Controls randomness and creativity of LLM model's response. Default: 0.2",
        ),
    ] = 0.2

    top_p: Annotated[
        float, Field(ge=0.1, le=1, description="Predictablity of LLM output.")
    ]
    max_output_token: Annotated[
        int, Field(ge=1, description="Maximum tokens of LLM output.")
    ]
    model_name: Annotated[str, Field(description="Name of LLM model used from Groq.")]


class Settings(BaseSettings):
    """Global settings parsed from environment variables."""

    groq_api_key: Annotated[
        SecretStr, Field(description="API key for Groq LLM provider.")
    ]
    vector_store: VectorStoreSettings
    search: SearchSettings
    llm: LLMSettings
    ingest: IngestSettings

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=ENV_FILE,
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key_prefix(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().startswith("gsk_"):
            raise PydanticCustomError(
                "value_error", "Groq API key must start with 'gsk_' prefix."
            )

        return v


settings = Settings()  # pyright: ignore[reportCallIssue]
"""Singleton setting pattern"""
