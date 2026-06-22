from datetime import datetime
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field

NonEmptyStr: TypeAlias = Annotated[str, Field(min_length=1)]
"""String type but cannot be empty ('')."""


class QueryMetadataFilter(BaseModel):
    """Structured metadata constraints extracted by the LLM for vector search filtering."""

    filename: Annotated[
        NonEmptyStr | None,
        Field(description="Restrict search to chunks from this exact filename."),
    ] = None

    uploaded_after: Annotated[
        datetime | None,
        Field(
            description="Only include chunks from documents uploaded after this UTC datetime."
        ),
    ] = None

    uploaded_before: Annotated[
        datetime | None,
        Field(
            description="Only include chunks from documents uploaded before this UTC datetime."
        ),
    ] = None


class VectorQuery(BaseModel):
    """LLM-generated search query and optional metadata filters for vector retrieval."""

    query: Annotated[
        NonEmptyStr,
        Field(description="Optimised search string for vector similarity retrieval."),
    ]

    filters: Annotated[
        QueryMetadataFilter,
        Field(
            description="Structured metadata constraints to narrow the Qdrant search."
        ),
    ] = QueryMetadataFilter()
