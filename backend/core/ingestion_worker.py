import asyncio
import shutil
import tempfile
from pathlib import Path

from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core.chunking import TextChunker
from backend.core.database import get_session_factory
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


async def _update_job(
    session: AsyncSession,
    job: IngestionJob,
    **kwargs: object,
) -> None:
    for key, val in kwargs.items():
        setattr(job, key, val)
    session.add(instance=job)
    await session.commit()
    await session.refresh(instance=job)


async def run_ingestion_worker(
    queue: asyncio.Queue[str],
    file_store: dict[str, tuple[str, bytes]],
    vector_store: VectorStore,
) -> None:
    session_factory = get_session_factory()
    while True:
        job_id = await queue.get()
        temp_dir: str | None = None
        try:
            async with session_factory() as session:
                job = await session.get(IngestionJob, job_id)
                if job is None:
                    continue

                await _update_job(session, job, status="processing", progress=10)
                signal_progress(job_id)

                safe_filename, raw_bytes = file_store.pop(job_id)
                temp_dir = tempfile.mkdtemp()
                temp_file_path = Path(temp_dir) / safe_filename

                _ = temp_file_path.write_bytes(data=raw_bytes)

                ingestor = DocumentIngestor(file_path=temp_file_path)
                documents = await ingestor.ingest_async()

                if not documents:
                    await _update_job(
                        session,
                        job,
                        status="failed",
                        error="File did not contain any valid text data.",
                    )
                    signal_progress(job_id)
                    continue

                await _update_job(session, job, progress=40)
                signal_progress(job_id)

                chunker = TextChunker()
                text_chunks = await chunker.achunk_text(documents=documents)

                await _update_job(session, job, progress=70)
                signal_progress(job_id)

                _ = await vector_store.ainsert_docs(
                    documents=text_chunks, session_id=job.session_id
                )

                await _update_job(
                    session,
                    job,
                    status="done",
                    progress=100,
                    chunk_count=len(text_chunks),
                )
                signal_progress(job_id)

        except Exception as exc:
            try:
                async with session_factory() as session:
                    job = await session.get(IngestionJob, job_id)
                    if job is not None:
                        await _update_job(session, job, status="failed", error=str(exc))
                signal_progress(job_id)
            except Exception:
                pass
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            cleanup_event(job_id)
            queue.task_done()
