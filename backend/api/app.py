# pyright: reportAny=none

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import status
from fastapi.applications import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRouter

from backend.api.documents import documents_router, get_vector_store
from backend.api.messages import messages_router
from backend.core.chunking import (
    _get_cached_tokenizer,  # pyright: ignore[reportPrivateUsage]
)
from backend.core.config import settings
from backend.core.database import close_db, init_db
import backend.core.models  # pyright: ignore[reportUnusedImport]
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
    _ = await asyncio.to_thread(_get_cached_tokenizer)
    await init_db()
    yield
    await vector_store.aclose()
    await close_db()


# Routers
main_api = APIRouter(prefix="/api")
main_api.include_router(router=documents_router)
main_api.include_router(router=messages_router)

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
                            "detail": f"File size exceeds the {settings.ingest.max_file_size}MB limit.",
                            "error_code": "FILE_TOO_LARGE",
                            "meta": {
                                "max_bytes": max_bytes,
                                "received_bytes": content_length,
                            },
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "detail": "Invalid Content-Length header format.",
                        "error_code": "INVALID_HEADER",
                    },
                )
    return await call_next(request)


def custom_openapi() -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Generates schema containing error dictionaries for middleware constraints."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Backend API",
        version="1.0.0",
        description="API Documentation featuring automated file size validation schemas.",
        routes=app.routes,
    )

    middleware_schemas = {
        "HTTPErrorResponse400": {
            "title": "HTTPErrorResponse400",
            "type": "object",
            "properties": {
                "detail": {
                    "title": "Detail",
                    "type": "string",
                    "example": "Invalid Content-Length header format.",
                },
                "error_code": {
                    "title": "Error Code",
                    "type": "string",
                    "example": "INVALID_HEADER",
                },
            },
            "required": ["detail", "error_code"],
        },
        "HTTPErrorResponse413": {
            "title": "HTTPErrorResponse413",
            "type": "object",
            "properties": {
                "detail": {
                    "title": "Detail",
                    "type": "string",
                    "example": f"File size exceeds the {settings.ingest.max_file_size}MB limit.",
                },
                "error_code": {
                    "title": "Error Code",
                    "type": "string",
                    "example": "FILE_TOO_LARGE",
                },
                "meta": {
                    "title": "Metadata",
                    "type": "object",
                    "properties": {
                        "max_bytes": {
                            "type": "integer",
                            "example": settings.ingest.max_file_size * 1024 * 1024,
                        },
                        "received_bytes": {"type": "integer", "example": 15000000},
                    },
                    "required": ["max_bytes", "received_bytes"],
                },
            },
            "required": ["detail", "error_code", "meta"],
        },
    }

    components = openapi_schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    schemas.update(middleware_schemas)

    middleware_responses = {
        "400": {
            "description": "Bad Request - Header validation failed.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/HTTPErrorResponse400"}
                }
            },
        },
        "413": {
            "description": "Request Entity Too Large - Upload limit breached.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/HTTPErrorResponse413"}
                }
            },
        },
    }

    for _, methods in openapi_schema.get("paths", {}).items():
        if "post" in methods:
            responses = methods["post"].setdefault("responses", {})
            for code, structure in middleware_responses.items():
                if code not in responses:
                    responses[code] = structure

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
