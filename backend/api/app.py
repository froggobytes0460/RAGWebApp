import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import status
from fastapi.applications import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRouter

from backend.api.documents import documents_router, get_vector_store
from backend.core.config import settings
from backend.core.vector_store import (
    _get_huggingface_embeddings,  # pyright: ignore[reportPrivateUsage]
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,  # pyright: ignore[reportUnusedParameter]
) -> AsyncGenerator[None]:
    vector_store = get_vector_store()
    _ = await vector_store.ainit_collection()
    _ = await asyncio.to_thread(_get_huggingface_embeddings)
    yield
    await vector_store.aclose()


# Routers:
main_api = APIRouter(prefix="/api")
main_api.include_router(router=documents_router)

# App initialization
app = FastAPI(lifespan=lifespan)
app.include_router(router=main_api)


@app.middleware(middleware_type="http")
async def limit_upload_size(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Limits filesize to `settings.ingest.max_file_size` via Content-Length header."""

    if request.method == "POST":
        content_length_str = request.headers.get("content-length")
        if content_length_str:
            try:
                content_length = int(content_length_str)
                max_bytes = settings.ingest.max_file_size * 1024 * 1024
                if content_length > max_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "detail": f"File size exceeds the {settings.ingest.max_file_size}MB limit."
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid Content-Length header format."},
                )
    return await call_next(request)
