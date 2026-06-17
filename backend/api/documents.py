from pathlib import Path
import shutil
import tempfile
from typing import cast

import aiofiles
from anyio.to_thread import run_sync
from fastapi import APIRouter, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi_utils.cbv import cbv
from werkzeug.utils import secure_filename

from backend.api.schemas import DocumentDeleteResponse, IngestResponse
from backend.core.chunking import TextChunker
from backend.core.config import settings
from backend.core.ingest import DocumentIngestor
from backend.core.vector_store import VectorStore

documents_router = APIRouter(
    prefix="/v1/chats/{session_id}/documents",
    tags=["Documents"],
    redirect_slashes=False,
)

_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore.from_settings()
    return _vector_store_instance


@cbv(router=documents_router)
class DocumentView:
    vector_store: VectorStore = cast(VectorStore, Depends(dependency=get_vector_store))

    @documents_router.post(path="/", status_code=status.HTTP_201_CREATED)
    async def create_document(
        self, session_id: str, file: UploadFile
    ) -> IngestResponse:
        """Upload, validate, and chunk a document into a dedicated chat session store."""

        file_suffix = Path(file.filename or "").suffix.lower()

        if file_suffix not in DocumentIngestor.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{file_suffix}'. Allowed formats: {', '.join(DocumentIngestor.allowed_extensions)}",
            )

        temp_dir = await run_sync(func=tempfile.mkdtemp)
        safe_filename = (
            secure_filename(filename=file.filename or "uploaded_file")
            or "uploaded_file"
        )
        if not safe_filename.endswith(file_suffix):
            safe_filename = Path(safe_filename).stem + file_suffix
        temp_file_path = Path(temp_dir) / safe_filename

        max_bytes = settings.ingest.max_file_size * 1024 * 1024
        bytes_read = 0
        try:
            async with aiofiles.open(temp_file_path, "wb") as buffer:
                while chunk := await file.read(64 * 1024):
                    bytes_read += len(chunk)
                    if bytes_read > max_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            detail=f"File content exceeded the {settings.ingest.max_file_size}MB limit during upload.",
                        )
                    _ = await buffer.write(chunk)

            ingestor = DocumentIngestor(file_path=temp_file_path)
            chunker = TextChunker()
            documents = await ingestor.ingest_async()

            if not documents:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The provided file did not contain any valid text data to parse.",
                )

            text_chunks = await chunker.achunk_text(documents=documents)
            _ = await self.vector_store.ainsert_docs(
                documents=text_chunks, session_id=session_id
            )

            return IngestResponse(
                document_id=safe_filename or "unknown_id", doc_count=len(text_chunks)
            )

        finally:
            await run_sync(shutil.rmtree, temp_dir)

    @documents_router.get(path="/")
    async def list_documents(self, session_id: str) -> list[dict[str, str]]:
        """Fetch all unique document files and their metadata uploaded inside this session."""

        return await self.vector_store.alist_documents(session_id=session_id)

    @documents_router.delete("/{filename}", status_code=status.HTTP_202_ACCEPTED)
    async def delete_document(
        self, session_id: str, filename: str
    ) -> DocumentDeleteResponse:
        """Purge all text embeddings associated with a specific file inside this chat."""

        del_result = await self.vector_store.adelete_document(
            session_id=session_id, filename=filename
        )
        return DocumentDeleteResponse(status=del_result.status)
