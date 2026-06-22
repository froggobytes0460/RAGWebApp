import asyncio
from functools import lru_cache
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Request, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.responses import StreamingResponse
from fastapi_utils.cbv import cbv
from werkzeug.utils import secure_filename

from backend.api.limiter import limiter
from backend.api.schemas import (
    DocumentDeleteResponse,
    IngestJobResponse,
    JobProgressResponse,
)
from backend.api.state import TypedFastAPI
from backend.core.config import settings
from backend.core.database import get_session_factory
from backend.core.ingest import DocumentIngestor
from backend.core.ingestion_worker import get_or_create_event
from backend.core.models import IngestionJob
from backend.core.vector_store import VectorStore

documents_router = APIRouter(
    prefix="/v1/chats/{session_id}/documents",
    tags=["Documents"],
    redirect_slashes=False,
)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore.from_settings()


@cbv(router=documents_router)
class DocumentView:
    vector_store: VectorStore = cast(VectorStore, Depends(dependency=get_vector_store))

    @documents_router.post(path="/", status_code=status.HTTP_202_ACCEPTED)
    @limiter.limit(  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
        limit_value="10/minute"
    )
    async def create_document(
        self,
        request: Request,
        session_id: str,
        file: UploadFile,
    ) -> IngestJobResponse:
        """Accept a document upload and enqueue it for background ingestion."""

        file_suffix = Path(file.filename or "").suffix.lower()

        if file_suffix not in DocumentIngestor.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{file_suffix}'. Allowed formats: {', '.join(DocumentIngestor.allowed_extensions)}",
            )

        safe_filename = (
            secure_filename(filename=file.filename or "uploaded_file")
            or "uploaded_file"
        )
        if not safe_filename.endswith(file_suffix):
            safe_filename = Path(safe_filename).stem + file_suffix

        max_bytes = settings.ingest.max_file_size * 1024 * 1024
        raw_bytes = b""
        while chunk := await file.read(64 * 1024):
            raw_bytes += chunk
            if len(raw_bytes) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File content exceeded the {settings.ingest.max_file_size}MB limit during upload.",
                )

        session_factory = get_session_factory()
        async with session_factory() as db:
            job = IngestionJob(session_id=session_id, filename=safe_filename)
            db.add(instance=job)
            await db.commit()
            await db.refresh(instance=job)

        typed_app = cast(TypedFastAPI, request.app)
        typed_app.typed_state.file_store[job.id] = (safe_filename, raw_bytes)
        await typed_app.typed_state.job_queue.put(item=job.id)

        return IngestJobResponse(job_id=job.id)

    @documents_router.get(path="/jobs/{job_id}/progress")
    async def stream_job_progress(
        self,
        request: Request,
        session_id: str,  # pyright: ignore[reportUnusedParameter]
        job_id: str,
    ) -> StreamingResponse:
        """SSE stream emitting ingestion progress events for a specific job."""

        session_factory = get_session_factory()

        async def _sse_generator():
            event = get_or_create_event(job_id)
            while True:
                if await request.is_disconnected():
                    break

                async with session_factory() as db:
                    job = await db.get(entity=IngestionJob, ident=job_id)

                if job is None:
                    yield 'event: error\ndata: {"error": "Job not found"}\n\n'
                    break

                payload = JobProgressResponse(
                    job_id=job.id,
                    filename=job.filename,
                    status=job.status,
                    progress=job.progress,
                    chunk_count=job.chunk_count,
                    error=job.error,
                )
                yield f"event: progress\ndata: {payload.model_dump_json(exclude_none=True)}\n\n"

                if job.status in ("done", "failed"):
                    break

                try:
                    _ = await asyncio.wait_for(fut=event.wait(), timeout=30)
                    event.clear()
                except asyncio.TimeoutError:
                    pass

        return StreamingResponse(
            content=_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @documents_router.get(path="/")
    async def list_documents(self, session_id: str) -> list[dict[str, str]]:
        """Fetch all unique document files and their metadata uploaded inside this session."""

        return await self.vector_store.alist_documents(session_id=session_id)

    @documents_router.delete(path="/{filename}", status_code=status.HTTP_202_ACCEPTED)
    async def delete_document(
        self, session_id: str, filename: str
    ) -> DocumentDeleteResponse:
        """Purge all text embeddings associated with a specific file inside this chat."""

        del_result = await self.vector_store.adelete_document(
            session_id=session_id, filename=filename
        )
        return DocumentDeleteResponse(status=del_result.status)
