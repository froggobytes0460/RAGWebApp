# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

- **Setup environment**: `uv sync`
- **Install pre‑commit hooks**: `uv run pre-commit install`
- **Run all tests**: `uv run pytest -q --tb=short --no-header`
- **Run a single test**: `uv run pytest tests/test_file.py::test_name`
- **Run pre‑commit checks**: `uv run pre-commit run --all-files`
- **Type‑check only**: `uv run pre-commit run basedpyright --all-files`
- **Start the API server**: `NO_COLOR=1 uv run fastapi dev backend/api > api.log 2>&1 &`
- **Stop the API server**: `pkill -f "fastapi dev"`

### Development Server (API)

- ALWAYS append `&` at the end of the command to start the dev server. The workflow must not stop due to continuous logging.
- ALWAYS prepend `NO_COLOR=1` to ensure the output written to the log file is stripped of raw ANSI styling code.
- ALWAYS redirect the output of the dev server command to `api.log`, combining standard output and error streams using `2>&1`.
- ALWAYS run `sleep 5 && tail -n 20 api.log` immediately after starting the server to verify it is running on the expected port.
- ALWAYS check if a server is already running on the target port before attempting a fresh start.
- ALWAYS stop the dev server when your tasks are completed.

## Architecture

The app is an async Python RAG pipeline exposed over HTTP via FastAPI. Request flow: `POST /v1/chats/{session_id}/documents` → `DocumentView` (CBV router) → `DocumentIngestor` → `TextChunker` → `VectorStore`.

### Layers

**API** (`backend/api/`)

- `app.py` – FastAPI app with an upload-size guard middleware (checks `Content-Length` *and* streams defensively).
- `documents.py` – `DocumentView` class-based view (via `fastapi-cbv`) on prefix `/v1/chats/{session_id}/documents`. Handles upload, list, and delete. `VectorStore` is injected via `Depends(VectorStore.from_settings)`.
- `schemas.py` – Pydantic models: `IngestResponse`, `MessageRequest` (with `top_k`, `score_threshold`, `filters`), `MessageResponse`, `RetrievedChunk`, `MetadataFilter`.

**Core** (`backend/core/`)

- `config.py` – `Settings` (Pydantic `BaseSettings`, singleton `settings`). Nested sub-settings: `IngestSettings`, `VectorStoreSettings`, `TextChunkSettings`, `SearchSettings`, `LLMSettings`. Uses `env_nested_delimiter="__"` so e.g. `VECTOR_STORE__COLLECTION_NAME` maps to `settings.vector_store.collection_name`. Groq and Qdrant API keys are validated against strict regex patterns.
- `ingest.py` – `DocumentIngestor` (Pydantic `BaseModel`). Binary-head validation via `verify_file_integrity` runs before Docling parses (guards against extension spoofing). Supported formats: `.pdf`, `.docx`, `.md`, `.xlsx`. Produces `Document` objects with `filename` and `page_number` in metadata.
- `chunking.py` – `TextChunker`. Uses `RecursiveCharacterTextSplitter` with a HuggingFace tokenizer (`BAAI/bge-large-en-v1.5` by default). Tokenizer and splitter are module-level singletons (thread-safe via `lru_cache` + `threading.Lock`).
- `vector_store.py` – `VectorStore` (Pydantic `BaseModel`). Wraps `QdrantVectorStore` (LangChain) with `QdrantClient` / `AsyncQdrantClient`. Supports local-path mode (sync-only fallback via `asyncio.to_thread`) and remote-URL mode (true async). Qdrant payload indexes on `metadata.session_id` (keyword) and `metadata.uploaded_at` (datetime) support scoped queries and TTL cleanup. Key methods: `ainit_collection`, `ainsert_docs`, `get_retriever`, `alist_documents`, `adelete_document`, `adelete_session`, `aclean_up_stale_vectors`.

### Key Invariants

- `vector_size` in `VectorStoreSettings` must match the output dimension of `embedding_model`. Default model (`all-MiniLM-L6-v2`) outputs 384-dim; default `vector_size` is 1024 — change both together.
- `TextChunkSettings.chunk_overlap` must be strictly less than `chunk_size` (enforced by model validator).
- `picture_classification` and `picture_description` toggles are silently forced off when `generate_picture_images=False`.
- File upload size is enforced twice: once via `Content-Length` header in middleware, once by streaming chunk accumulation in the route handler.

## Testing

Tests live in `tests/`. `conftest.py` provides session-scoped and function-scoped fixtures:

- `mock_qdrant_clients` – patches `QdrantClient` and `AsyncQdrantClient` at module level; avoids real network calls.
- `vector_store` – parametrized over `local_path` and `remote_url` modes, injects `MockEmbeddings` to avoid downloading HuggingFace models.
- `reset_tokenizer_cache` (autouse) – clears `lru_cache` and `TextChunker._recursive_text_splitter` before each test to prevent state bleed.

Coverage must stay ≥ 80% (`--cov-fail-under=80`). `asyncio_mode = "auto"` is set so async test functions need no decorator.

## Important Notes

- Pre‑commit enforces `black` formatting and `basedpyright` type checking; both must pass before a commit lands.
- `settings` is a module-level singleton instantiated at import time — use `monkeypatch.setattr` to override fields in tests rather than re-importing.
- When connecting to Qdrant Cloud (`*.qdrant.tech`), `QDRANT_API_KEY` is required and validated against a strict base64 pattern.
