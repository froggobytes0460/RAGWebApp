from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from pydantic.fields import Field
from pydantic.functional_validators import field_validator, model_validator
from pydantic.main import BaseModel
from pydantic.networks import AnyHttpUrl
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import re

BASE_DIR = Path(__file__).parents[2]
ENV_FILE = BASE_DIR / ".env"


class IngestSettings(BaseModel):
    max_file_size: Annotated[
        float, Field(gt=0, description="Maximum filesize for each file (in MB).")
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

    QDRANT_API_KEY_PATTERN: ClassVar[str] = r"^[A-Za-z0-9+/]{43,44}=?$"

    api_key: Annotated[
        SecretStr | None, Field(description="API key for connecting to Qdrant Cloud.")
    ] = None

    collection_name: Annotated[
        str,
        Field(
            description="The target collection segment name inside the Qdrant database instance."
        ),
    ] = "rag_documents"

    embedding_model: Annotated[
        str,
        Field(description="HuggingFace model ID used to compute text vector profiles."),
    ] = "sentence-transformers/all-MiniLM-L6-v2"

    normalize_embeddings: Annotated[
        bool,
        Field(
            description="Normalize text chunk embeddings before insertion into vector store."
        ),
    ] = True

    url_or_path: Annotated[
        AnyHttpUrl | Path,
        Field(
            description="URL or local path for Qdrant connection, default 'BASE_DIR/.qdrant_local/'."
        ),
    ] = (
        BASE_DIR / ".qdrant_local/"
    )

    ttl: Annotated[
        int,
        Field(
            ge=3600,
            description="TTL for embedding vectors stored in vectorstore in seconds, minimum is one hour. (default: 24 hours.)",
        ),
    ] = 86400

    vector_size: Annotated[
        int, Field(ge=384, description="Size of embedding vectors.")
    ] = 384

    @model_validator(mode="after")
    def validate_qdrant_auth(self) -> Self:
        url_or_path = self.url_or_path

        if not isinstance(url_or_path, Path):
            host = url_or_path.host or ""

            if "qdrant.tech" in host:
                if not self.api_key:
                    raise ValueError(
                        f"A 'qdrant_api_key' is required when connecting to Qdrant Cloud cluster ({host})."
                    )

                if not re.match(
                    self.QDRANT_API_KEY_PATTERN, self.api_key.get_secret_value()
                ):
                    raise ValueError(
                        "The provided 'qdrant_api_key' does not match the strict Qdrant Cloud pattern requirement."
                    )

        return self

    @field_validator("url_or_path")
    @classmethod
    def verify_vectorstore_path(cls, v: Path | AnyHttpUrl) -> Path | AnyHttpUrl:
        if not isinstance(v, Path):
            return v

        if not v.is_absolute():
            v = (BASE_DIR / v).resolve()

        if not v.is_file():
            v.mkdir(parents=True, exist_ok=True)
            return v

        raise ValueError("Vector store storage path cannot be a file.")


class TextChunkSettings(BaseModel):
    tokenizer_model: Annotated[
        str,
        Field(description="HuggingFace model ID used to tokenize."),
    ] = "BAAI/bge-large-en-v1.5"

    chunk_size: Annotated[int, Field(gt=0, description="Size of each chunk size.")]

    chunk_overlap: Annotated[
        int,
        Field(
            ge=0,
            description="Amount of tokens allowed to be overlapped between text chunks.",
        ),
    ]

    @model_validator(mode="after")
    def chunk_overlap_less_than_size(self) -> Self:
        if not self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
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
            description="Relevance confidence minimum; only used if search_type is set to `similarity_score_threshold`.",
        ),
    ] = 0.5

    lambda_mult: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Controls relevance for `mmr` search type.",
        ),
    ] = 0.5


class LLMSettings(BaseModel):
    """Settings for LLM."""

    api_key: Annotated[SecretStr, Field(description="API key for LLM provider.")]

    max_retries: Annotated[
        int, Field(description="Maximum retries on LLM output generation.", gt=0)
    ] = 3

    temperature: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Controls randomness and creativity of LLM model's response. Default: 0.2",
        ),
    ] = 0.2

    max_output_token: Annotated[
        int, Field(ge=1, description="Maximum tokens of LLM output.")
    ]
    model_name: Annotated[str, Field(description="Name of LLM model used from Groq.")]

    provider: Annotated[
        Literal["groq", "openrouter"],
        Field(description="LLM provider. Determines which client is instantiated."),
    ] = "groq"


class Settings(BaseSettings):
    """Global settings parsed from environment variables."""

    vector_store: VectorStoreSettings

    text_chunk: TextChunkSettings

    search: SearchSettings

    llm: LLMSettings

    ingest: IngestSettings

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=ENV_FILE,
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
"""Singleton setting pattern"""
