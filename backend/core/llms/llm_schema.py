from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field, StringConstraints

NonEmptyStr: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class HypeQuestions(BaseModel):
    """Questions to generate from text chunks for HyPE."""

    questions: Annotated[
        list[NonEmptyStr],
        Field(description="List of questions raised from text chunk."),
    ]
