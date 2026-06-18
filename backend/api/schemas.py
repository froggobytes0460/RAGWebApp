from datetime import datetime
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, Field, model_validator
from qdrant_client.models import UpdateStatus

NonEmptyStr: TypeAlias = Annotated[str, Field(min_length=1)]
"""String type but cannot be empty ('')."""

NonZeroOrNegativeInt: TypeAlias = Annotated[int, Field(gt=0)]
"""Int type, but value > 0 ."""


class MetadataFilter(BaseModel):
    """Structured metadata filters for scoping vector search queries."""

    filenames: Annotated[
        list[NonEmptyStr] | None,
        Field(
            description="Filter results to matches within specific source files.",
        ),
    ] = None
    page_min: Annotated[
        NonZeroOrNegativeInt | None,
        Field(
            description="Inclusive lower bound for the source page number.",
        ),
    ] = None
    page_max: Annotated[
        NonZeroOrNegativeInt | None,
        Field(
            description="Inclusive upper bound for the source page number.",
        ),
    ] = None

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_min and self.page_max and self.page_min > self.page_max:
            raise ValueError("page_min cannot be greater than page_max")
        return self


class IngestResponse(BaseModel):
    """
    Response model from ingestion endpoint.

    The target index/session context is managed via the URL path.
    """

    document_id: Annotated[
        NonEmptyStr,
        Field(
            description="The generated unique ID for the tracking document resource."
        ),
    ]
    doc_count: Annotated[
        NonZeroOrNegativeInt,
        Field(description="The count of text chunks/documents split and ingested."),
    ]


class MessageRequest(BaseModel):
    """Request model for adding a message to a specific chat resource."""

    question: Annotated[
        NonEmptyStr, Field(description="Question or prompt queried from the LLM.")
    ]

    top_k: Annotated[
        int,
        Field(
            gt=0,
            le=50,
            description="Number of context chunks to retrieve from the vector database.",
        ),
    ] = 4

    score_threshold: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum similarity score cutoff to filter out bad matches.",
        ),
    ] = None

    filters: Annotated[
        MetadataFilter | None,
        Field(
            description="Structured metadata filters passed directly to the Vector DB.",
        ),
    ] = None


class RetrievedChunk(BaseModel):
    """The full chunk payload returned from the vector search engine."""

    content: Annotated[
        NonEmptyStr, Field(description="Actual raw text content of the chunk.")
    ]
    score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Vector search similarity matching confidence score.",
        ),
    ]
    filename: Annotated[
        NonEmptyStr, Field(description="File from which the text chunk originates.")
    ]
    page_number: (
        Annotated[
            NonZeroOrNegativeInt,
            Field(description="Page number the text chunk was extracted from."),
        ]
        | None
    ) = None


class DocumentDeleteResponse(BaseModel):
    status: Annotated[
        UpdateStatus,
        Field(
            description="The synchronization or execution status of the delete operation",
        ),
    ]


class StreamChunk(BaseModel):
    """SSE payload shapes for the streaming message endpoint."""

    text: Annotated[
        str,
        Field(description="LLM token chunk emitted during streaming."),
    ] = ""
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Populated only in the final 'done' event."
    )


class MessageHistoryItem(BaseModel):
    """A single message entry returned from GET /messages history."""

    id: Annotated[
        int | None,
        Field(description="Database primary key of the message row."),
    ]
    role: Annotated[
        Literal["user", "ai"],
        Field(description="Who sent this message."),
    ]
    content: Annotated[
        NonEmptyStr,
        Field(description="Full text of the message."),
    ]
    created_at: Annotated[
        datetime,
        Field(description="UTC timestamp when the message was persisted."),
    ]
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
