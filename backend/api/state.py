import asyncio
from dataclasses import dataclass, field

from fastapi.applications import FastAPI
from slowapi import Limiter


@dataclass
class AppState:
    job_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    file_store: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    limiter: Limiter | None = None


class TypedFastAPI(FastAPI):
    @property
    def typed_state(self) -> AppState:
        return self.state._typed  # pyright: ignore[reportAttributeAccessIssue]

    @typed_state.setter
    def typed_state(self, value: AppState) -> None:
        self.state._typed = value  # pyright: ignore[reportAttributeAccessIssue]
