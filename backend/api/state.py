import asyncio
from typing import ClassVar

from fastapi.applications import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter


class AppState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    job_queue: asyncio.Queue[str] = Field(
        description="Ingestion job queue.", default_factory=asyncio.Queue
    )
    file_store: dict[str, tuple[str, bytes]] = Field(default_factory=dict)
    limiter: Limiter | None = None


class TypedFastAPI(FastAPI):
    @property
    def typed_state(self) -> AppState:
        return self.state._typed  # pyright: ignore[reportAny]

    @typed_state.setter
    def typed_state(self, value: AppState) -> None:
        self.state._typed = value
