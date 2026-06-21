# pyright: reportExplicitAny=none
# pyright: reportIncompatibleVariableOverride=none

from datetime import datetime, timezone
from typing import Any, ClassVar, Literal
import uuid

import sqlalchemy as sa
from sqlmodel import Column, Field, SQLModel, String


class ChatSession(SQLModel, table=True):
    __tablename__: ClassVar[str] = "chat_sessions"
    __table_args__: ClassVar[dict[str, Any]] = {"extend_existing": True}

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        sa_column=Column(
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


class ChatMessage(SQLModel, table=True):
    __tablename__: ClassVar[str] = "chat_messages"
    __table_args__: ClassVar[dict[str, Any]] = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(
        foreign_key="chat_sessions.id",
        index=True,
    )
    role: Literal["user", "ai"] = Field(
        sa_column=Column(sa.String(length=4), nullable=False)
    )
    content: str = Field(sa_column=Column(sa.Text, nullable=False))
    retrieved_chunks: list[dict[str, Any]] | None = Field(
        default=None,
        sa_column=Column(sa.JSON, nullable=True),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        sa_column=Column(
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


class IngestionJob(SQLModel, table=True):
    __tablename__: ClassVar[str] = "ingestion_jobs"
    __table_args__: ClassVar[dict[str, Any]] = {"extend_existing": True}

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    session_id: str = Field(index=True)
    filename: str
    status: Literal["queued", "processing", "done", "failed"] = Field(
        default="queued",
        sa_column=Column(String, nullable=False),
    )
    progress: int = Field(default=0)
    chunk_count: int | None = Field(default=None)
    error: str | None = Field(default=None, sa_column=Column(sa.Text, nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        sa_column=Column(
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
