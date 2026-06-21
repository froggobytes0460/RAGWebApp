from pathlib import Path
import re
from typing import Annotated, ClassVar, Literal, Self

from pydantic.fields import Field
from pydantic.functional_validators import field_validator, model_validator
from pydantic.main import BaseModel
from pydantic.networks import AnyHttpUrl, AnyUrl, PostgresDsn, UrlConstraints
from pydantic.types import FilePath, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SqliteDsn = Annotated[
    AnyUrl,
    UrlConstraints(
        allowed_schemes=["sqlite+aiosqlite"],
        host_required=False,
    ),
]

BASE_DIR = Path(__file__).parents[2]
ENV_FILE = BASE_DIR / ".env"


class LogSettings(BaseModel):
    level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(description="Log level."),
    ] = "INFO"
    file_path: Path | None = BASE_DIR / "logs/app.logs"
    max_bytes: Annotated[int, Field(gt=0)] = 10 * 1024 * 1024
    backup_count: Annotated[int, Field(ge=0)] = 5

    @field_validator("file_path", mode="after")
    @classmethod
    def ensure_file_path_resolved(cls, v: Path | None) -> Path | None:
        if not v:
            return v

        resolved_path = v.resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        return resolved_path


class IngestSettings(BaseModel):
    max_file_size: Annotated[
        float, Field(gt=0, description="Maximum filesize for each file (in MB).")
    ] = 50

    # PDF options
    pdf_extract_images: Annotated[
        bool,
        Field(description="Attempt to extract text embedded in PDF images via pypdf."),
    ] = False

    # DOCX options
    docx_include_headers_footers: Annotated[
        bool,
        Field(
            description="Include text from header and footer sections of DOCX files."
        ),
    ] = False

    docx_include_tables: Annotated[
        bool,
        Field(description="Extract and include text from tables in DOCX files."),
    ] = True

    # XLSX options
    xlsx_include_empty_rows: Annotated[
        bool,
        Field(description="Include rows where all cells are empty in XLSX output."),
    ] = False


class VectorStoreSettings(BaseModel):
    """Settings for vector store."""

    QDRANT_API_KEY_PATTERN: ClassVar[str] = r"^[A-Za-z0-9+/_-]{43,44}=?$"

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
        Field(description="FastEmbed model name used to compute text vector profiles."),
    ] = "BAAI/bge-small-en-v1.5"

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

    prefer_qdrant_grpc: Annotated[
        bool, Field(description="Prefer using Qdrant gRPC instead of RestAPI.")
    ] = True

    grpc_port: Annotated[
        int, Field(ge=1, le=65535, description="Qdrant gRPC port.")
    ] = 6334

    tls_ca_cert: Annotated[
        FilePath | None,
        Field(
            description="Path to CA cert for self-signed Qdrant TLS. None uses system CAs."
        ),
    ] = None

    @model_validator(mode="after")
    def validate_qdrant_auth(self) -> Self:
        url_or_path = self.url_or_path

        if not isinstance(url_or_path, Path):
            host = url_or_path.host or ""

            if "cloud.qdrant.io" in host:
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
        Field(
            description="Model ID used to tokenize text chunks (via tokenizers library)."
        ),
    ] = "BAAI/bge-small-en-v1.5"

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


class DatabaseSettings(BaseModel):
    """Settings for the relational database (SQLite or PostgreSQL)."""

    uri: Annotated[
        SqliteDsn | PostgresDsn,
        Field(
            description=(
                "Async SQLAlchemy connection string. "
                "SQLite default: 'sqlite+aiosqlite:///./rag.db'. "
                "PostgreSQL example: 'postgresql+asyncpg://user:pass@host/db'."
            )
        ),
    ] = "sqlite+aiosqlite:///./rag.db"  # pyright: ignore[reportAssignmentType]

    echo_sql: Annotated[
        bool,
        Field(description="Log all SQL statements to stdout. Useful for debugging."),
    ] = False

    @field_validator("uri")
    @classmethod
    def validate_url(cls, v: SqliteDsn | PostgresDsn) -> SqliteDsn | PostgresDsn:
        scheme = str(v).split("://")[0]
        if scheme == "sqlite+aiosqlite":
            return v
        if scheme != "postgresql+asyncpg":
            raise ValueError(
                f"PostgreSQL URLs must use the 'postgresql+asyncpg' driver, got '{scheme}'."
            )
        return v


class Settings(BaseSettings):
    """Global settings parsed from environment variables."""

    log: LogSettings = LogSettings()

    database: DatabaseSettings = DatabaseSettings()

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
