from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic.fields import Field
from pydantic.functional_validators import field_validator, model_validator
from pydantic.main import BaseModel
from pydantic.types import DirectoryPath, SecretStr
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parents[2]
ENV_FILE = BASE_DIR / ".env"


class VectorStoreSettings(BaseModel):
    """Settings for vector store."""

    path: DirectoryPath = Field(
        default=BASE_DIR / "chroma_db",
        description="Local directory path where Chroma DB will persist its data files.",
    )
    collection_name: str = Field(
        default="rag_documents",
        description="The target collection segment name inside the Chroma database instance.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID or Ollama name used to compute text vector profiles.",
    )
    chunk_size: int = Field(..., gt=0, description="Size of each chunk size.")
    chunk_overlap: int = Field(..., ge=0, description="Amount of tokens allowed to be ")

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

    top_k: int = Field(
        ..., ge=1, description="Number of best-matching chunks for system retrieval."
    )
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        default="similarity",
        description="The strategy algorithm LangChain employs to pull related reference context.",
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance confidence minimum; only used if search_type is set to threshold matching.",
    )


class LLMSettings(BaseModel):
    """Settings for LLM."""

    temperature: float = Field(
        ge=0,
        le=1,
        default=0.2,
        description="Controls randomness and creativity of LLM model's response. Default: 0.2",
    )
    top_p: float = Field(..., ge=0.1, le=1, description="Predictablity of LLM output.")
    max_output_token: int = Field(
        ..., ge=1, description="Maximum tokens of LLM output."
    )
    model_name: str = Field(..., description="Name of LLM model used from Groq.")


class Settings(BaseSettings):
    """Global settings parsed from environment variables."""

    groq_api_key: SecretStr = Field(..., description="API key for Groq LLM provider.")
    vector_store: VectorStoreSettings
    search: SearchSettings
    llm: LLMSettings

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
