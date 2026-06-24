from pydantic import BaseModel


class HypeQuestions(BaseModel):
    """Questions to generate from text chunks for HyPE."""

    questions: list[str]
