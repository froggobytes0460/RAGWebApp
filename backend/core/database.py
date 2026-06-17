from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core.config import settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    return create_async_engine(
        url=settings.database.url,
        echo=settings.database.echo_sql,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker[AsyncSession](
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def init_db() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(fn=SQLModel.metadata.create_all)


async def close_db() -> None:
    await get_engine().dispose()


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with get_session_factory()() as session:
        yield session
