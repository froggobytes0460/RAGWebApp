# pyright: reportAny=none

import asyncio

import anyio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
import pytest

from backend.core.ingestion_worker import (
    cleanup_event,
    get_or_create_event,
    run_ingestion_worker,
    signal_progress,
)
from backend.core.models import IngestionJob


def _make_mock_llm_client() -> MagicMock:
    mock = MagicMock()
    mock.generate_hype_questions = AsyncMock(
        return_value=["What is X?", "How does Y work?", "When did Z occur?"]
    )
    return mock


@pytest.fixture
def make_session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker[AsyncSession](
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )


class TestEventHelpers:
    def test_get_or_create_event_creates_new(self) -> None:
        job_id = "evt-new-1"
        cleanup_event(job_id)
        event = get_or_create_event(job_id)
        assert isinstance(event, anyio.Event)
        cleanup_event(job_id)

    def test_get_or_create_event_returns_same(self) -> None:
        job_id = "evt-same-1"
        cleanup_event(job_id)
        e1 = get_or_create_event(job_id)
        e2 = get_or_create_event(job_id)
        assert e1 is e2
        cleanup_event(job_id)

    def test_signal_progress_sets_event(self) -> None:
        job_id = "evt-sig-1"
        cleanup_event(job_id)
        event = get_or_create_event(job_id)
        assert not event.is_set()
        signal_progress(job_id)
        assert event.is_set()
        cleanup_event(job_id)

    def test_signal_progress_missing_job_noop(self) -> None:
        signal_progress("nonexistent-job-id")

    def test_cleanup_removes_event(self) -> None:
        job_id = "evt-clean-1"
        _ = get_or_create_event(job_id)
        cleanup_event(job_id)
        signal_progress(job_id)  # should be a noop without crashing


class TestRunIngestionWorker:
    async def test_successful_ingestion(
        self, make_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        parsed_doc = Document(
            page_content="hello world",
            metadata={"filename": "test.pdf", "page_number": 1},
        )
        mock_vs = MagicMock()
        mock_vs.ainsert_hype_docs = AsyncMock(return_value=["id1", "id2", "id3"])
        mock_llm = _make_mock_llm_client()

        async with make_session_factory() as session:
            job = IngestionJob(session_id="sess-1", filename="test.pdf")
            session.add(job)
            await session.commit()
            await session.refresh(job)
            job_id = job.id

        queue: asyncio.Queue[str] = asyncio.Queue()
        file_store: dict[str, tuple[str, bytes]] = {job_id: ("test.pdf", b"%PDF fake")}

        with (
            patch(
                "backend.core.ingestion_worker.get_session_factory",
                return_value=make_session_factory,
            ),
            patch(
                "backend.core.ingestion_worker.DocumentIngestor.ingest_async",
                new=AsyncMock(return_value=[parsed_doc]),
            ),
            patch(
                "backend.core.ingestion_worker.TextChunker.achunk_text",
                new=AsyncMock(return_value=[parsed_doc, parsed_doc]),
            ),
            patch(
                "backend.core.ingestion_worker.LLMClientFactory.from_settings",
                return_value=mock_llm,
            ),
        ):
            await queue.put(job_id)
            worker = asyncio.create_task(
                run_ingestion_worker(
                    queue=queue, file_store=file_store, vector_store=mock_vs
                )
            )
            await queue.join()
            _ = worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        async with make_session_factory() as session:
            updated = await session.get(IngestionJob, job_id)
            assert updated is not None
            assert updated.status == "done"
            assert updated.progress == 100
            assert updated.chunk_count == 2
        mock_vs.ainsert_hype_docs.assert_called_once()

    async def test_hype_generation_failure_falls_back(
        self, make_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        parsed_doc = Document(
            page_content="fallback content",
            metadata={"filename": "test.pdf", "page_number": 1},
        )
        mock_vs = MagicMock()
        mock_vs.ainsert_hype_docs = AsyncMock(return_value=["id1"])
        mock_llm = MagicMock()
        mock_llm.generate_hype_questions = AsyncMock(return_value=[])

        async with make_session_factory() as session:
            job = IngestionJob(session_id="sess-fallback", filename="test.pdf")
            session.add(job)
            await session.commit()
            await session.refresh(job)
            job_id = job.id

        queue: asyncio.Queue[str] = asyncio.Queue()
        file_store: dict[str, tuple[str, bytes]] = {job_id: ("test.pdf", b"%PDF fake")}

        with (
            patch(
                "backend.core.ingestion_worker.get_session_factory",
                return_value=make_session_factory,
            ),
            patch(
                "backend.core.ingestion_worker.DocumentIngestor.ingest_async",
                new=AsyncMock(return_value=[parsed_doc]),
            ),
            patch(
                "backend.core.ingestion_worker.TextChunker.achunk_text",
                new=AsyncMock(return_value=[parsed_doc]),
            ),
            patch(
                "backend.core.ingestion_worker.LLMClientFactory.from_settings",
                return_value=mock_llm,
            ),
        ):
            await queue.put(job_id)
            worker = asyncio.create_task(
                run_ingestion_worker(
                    queue=queue, file_store=file_store, vector_store=mock_vs
                )
            )
            await queue.join()
            _ = worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        async with make_session_factory() as session:
            updated = await session.get(IngestionJob, job_id)
            assert updated is not None
            assert updated.status == "done"
        mock_vs.ainsert_hype_docs.assert_called_once()

    async def test_empty_parse_marks_failed(
        self, make_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        mock_vs = MagicMock()

        async with make_session_factory() as session:
            job = IngestionJob(session_id="sess-2", filename="empty.pdf")
            session.add(job)
            await session.commit()
            await session.refresh(job)
            job_id = job.id

        queue: asyncio.Queue[str] = asyncio.Queue()
        file_store: dict[str, tuple[str, bytes]] = {job_id: ("empty.pdf", b"%PDF fake")}

        with (
            patch(
                "backend.core.ingestion_worker.get_session_factory",
                return_value=make_session_factory,
            ),
            patch(
                "backend.core.ingestion_worker.DocumentIngestor.ingest_async",
                new=AsyncMock(return_value=[]),
            ),
        ):
            await queue.put(job_id)
            worker = asyncio.create_task(
                run_ingestion_worker(
                    queue=queue, file_store=file_store, vector_store=mock_vs
                )
            )
            await queue.join()
            _ = worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        async with make_session_factory() as session:
            updated = await session.get(IngestionJob, job_id)
            assert updated is not None
            assert updated.status == "failed"
            assert updated.error is not None

    async def test_ingest_exception_marks_failed(
        self, make_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        mock_vs = MagicMock()

        async with make_session_factory() as session:
            job = IngestionJob(session_id="sess-3", filename="boom.pdf")
            session.add(job)
            await session.commit()
            await session.refresh(job)
            job_id = job.id

        queue: asyncio.Queue[str] = asyncio.Queue()
        file_store: dict[str, tuple[str, bytes]] = {job_id: ("boom.pdf", b"%PDF fake")}

        with (
            patch(
                "backend.core.ingestion_worker.get_session_factory",
                return_value=make_session_factory,
            ),
            patch(
                "backend.core.ingestion_worker.DocumentIngestor.ingest_async",
                new=AsyncMock(side_effect=RuntimeError("parse boom")),
            ),
        ):
            await queue.put(job_id)
            worker = asyncio.create_task(
                run_ingestion_worker(
                    queue=queue, file_store=file_store, vector_store=mock_vs
                )
            )
            await queue.join()
            _ = worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        async with make_session_factory() as session:
            updated = await session.get(IngestionJob, job_id)
            assert updated is not None
            assert updated.status == "failed"
            assert "parse boom" in (updated.error or "")
