import asyncio
import tempfile
from pathlib import Path

from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core.chunking import TextChunker
from backend.core.database import get_session_factory
from backend.core.logging import app_logger
from backend.core.ingest import DocumentIngestor
from backend.core.models import IngestionJob
from backend.core.vector_store import VectorStore

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


async def _update_job_and_signal(
    session: AsyncSession,
    job: IngestionJob,
    job_id: str,
    **kwargs: object,
) -> None:
    for key, val in kwargs.items():
        setattr(job, key, val)
    session.add(instance=job)
    await session.commit()
    await session.refresh(instance=job)
    signal_progress(job_id)


async def run_ingestion_worker(
    queue: asyncio.Queue[str],
    file_store: dict[str, tuple[str, bytes]],
    vector_store: VectorStore,
) -> None:
    session_factory = get_session_factory()
    while True:
        job_id = await queue.get()
        try:
            async with session_factory() as session:
                job = await session.get(IngestionJob, job_id)
                if job is None:
                    continue

                app_logger.job_started(job_id, job.filename)
                await _update_job_and_signal(
                    session, job, job_id, status="processing", progress=10
                )

                safe_filename, raw_bytes = file_store.pop(job_id)
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = Path(temp_dir) / safe_filename
                    _ = temp_file_path.write_bytes(data=raw_bytes)

                    documents = await DocumentIngestor(
                        file_path=temp_file_path
                    ).ingest_async()

                if not documents:
                    app_logger.job_empty(job_id, safe_filename)
                    await _update_job_and_signal(
                        session,
                        job,
                        job_id,
                        status="failed",
                        error="File did not contain any valid text data.",
                    )
                    continue

                await _update_job_and_signal(session, job, job_id, progress=40)

                text_chunks = await TextChunker().achunk_text(documents=documents)
                app_logger.job_chunked(job_id, len(text_chunks))

                await _update_job_and_signal(session, job, job_id, progress=70)

                _ = await vector_store.ainsert_docs(
                    documents=text_chunks, session_id=job.session_id
                )

                app_logger.job_complete(job_id, len(text_chunks), job.session_id)
                await _update_job_and_signal(
                    session,
                    job,
                    job_id,
                    status="done",
                    progress=100,
                    chunk_count=len(text_chunks),
                )

        except Exception as exc:
            app_logger.job_failed("Ingestion failed for job %s", job_id)
            try:
                async with session_factory() as session:
                    job = await session.get(IngestionJob, job_id)
                    if job is not None:
                        await _update_job_and_signal(
                            session, job, job_id, status="failed", error=str(exc)
                        )
            except Exception:
                app_logger.job_failed(
                    "Failed to persist error state for job %s", job_id
                )
        finally:
            cleanup_event(job_id)
            queue.task_done()
