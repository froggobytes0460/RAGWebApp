import asyncio
from pathlib import Path
import tempfile
from typing import Literal

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core.chunking import TextChunker
from backend.core.database import get_session_factory
from backend.core.ingest import DocumentIngestor
from backend.core.logging import app_logger
from backend.core.models import IngestionJob
from backend.core.vector_store import VectorStore

# Maps job_id → Event so the SSE progress stream can be woken up after each DB write.
_progress_events: dict[str, asyncio.Event] = {}


def get_or_create_event(job_id: str) -> asyncio.Event:
    if job_id not in _progress_events:
        _progress_events[job_id] = asyncio.Event()
    return _progress_events[job_id]


def signal_progress(job_id: str) -> None:
    event = _progress_events.get(job_id)
    if event:
        event.set()


def cleanup_event(job_id: str) -> None:
    _ = _progress_events.pop(job_id, None)


async def _persist_job_update(
    job_id: str,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: Literal["queued", "processing", "done", "failed"] | None = None,
    progress: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    """Write job fields to the DB, then wake up the SSE progress stream."""
    async with session_factory() as session:
        job = await session.get(entity=IngestionJob, ident=job_id)
        if job is None:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if chunk_count is not None:
            job.chunk_count = chunk_count
        if error is not None:
            job.error = error
        session.add(instance=job)
        await session.commit()
    signal_progress(job_id)


async def _run_pipeline(
    job_id: str,
    session_id: str,
    safe_filename: str,
    raw_bytes: bytes,
    vector_store: VectorStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Parse the uploaded file, chunk it, and insert into the vector store."""
    await _persist_job_update(job_id, session_factory, status="processing", progress=10)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / safe_filename
        _ = temp_file.write_bytes(data=raw_bytes)
        documents = await DocumentIngestor(file_path=temp_file).ingest_async()

    if not documents:
        app_logger.job_empty(job_id, safe_filename)
        await _persist_job_update(
            job_id,
            session_factory,
            status="failed",
            error="File did not contain any valid text data.",
        )
        return

    await _persist_job_update(job_id, session_factory, progress=40)

    text_chunks = await TextChunker().achunk_text(documents=documents)
    app_logger.job_chunked(job_id, len(text_chunks))

    await _persist_job_update(job_id, session_factory, progress=70)

    _ = await vector_store.ainsert_docs(documents=text_chunks, session_id=session_id)

    app_logger.job_complete(job_id, len(text_chunks), session_id)
    await _persist_job_update(
        job_id,
        session_factory,
        status="done",
        progress=100,
        chunk_count=len(text_chunks),
    )


async def run_ingestion_worker(
    queue: asyncio.Queue[str],
    file_store: dict[str, tuple[str, bytes]],
    vector_store: VectorStore,
) -> None:
    """Consume job IDs from the queue and run the ingest pipeline for each one."""
    session_factory = get_session_factory()

    while True:
        job_id = await queue.get()
        try:
            async with session_factory() as session:
                job = await session.get(IngestionJob, job_id)
                if job is None:
                    _ = file_store.pop(job_id, None)
                    continue
                filename = job.filename
                session_id = job.session_id

            app_logger.job_started(job_id, filename)
            safe_filename, raw_bytes = file_store.pop(job_id)

            await _run_pipeline(
                job_id=job_id,
                session_id=session_id,
                safe_filename=safe_filename,
                raw_bytes=raw_bytes,
                vector_store=vector_store,
                session_factory=session_factory,
            )

        except Exception as exc:
            app_logger.job_failed("Ingestion failed for job %s", job_id)
            try:
                await _persist_job_update(
                    job_id, session_factory, status="failed", error=str(exc)
                )
            except Exception:
                app_logger.job_failed(
                    "Failed to persist error state for job %s", job_id
                )
                # Wake the SSE stream even if the DB write failed so it does not
                # hang indefinitely waiting for a status update that will never land.
                signal_progress(job_id)
        finally:
            cleanup_event(job_id)
            queue.task_done()
