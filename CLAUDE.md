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

### Frontend (`frontend/`)

- **Install deps**: `cd frontend && npm install`
- **Dev server** (proxies `/api` to `http://127.0.0.1:8000`): `cd frontend && npm run dev` → `http://localhost:5173`
- **Production build**: `cd frontend && npm run build` (outputs to `frontend/dist/`, served by FastAPI automatically)
- **Type-check + build**: `npm run build` runs `tsc -b` before Vite, so it catches type errors
- **Lint**: `cd frontend && npm run lint`

### Development Server (API)

- ALWAYS append `&` at the end of the command to start the dev server. The workflow must not stop due to continuous logging.
- ALWAYS prepend `NO_COLOR=1` to ensure the output written to the log file is stripped of raw ANSI styling code.
- ALWAYS redirect the output of the dev server command to `api.log`, combining standard output and error streams using `2>&1`.
- ALWAYS run `sleep 5 && tail -n 20 api.log` immediately after starting the server to verify it is running on the expected port.
- ALWAYS check if a server is already running on the target port before attempting a fresh start.
- ALWAYS stop the dev server when your tasks are completed.

## Architecture

The app is an async Python RAG pipeline exposed over HTTP via FastAPI, with a React SPA frontend (`frontend/`) served as static files in production.

**Document ingest flow:** `POST /v1/chats/{session_id}/documents` → `DocumentView` → `DocumentIngestor` → `TextChunker` → `VectorStore`

**Chat flow:** `POST /v1/chats/{session_id}/messages` → `MessageView` → Qdrant retrieval → LLM client → SSE stream back to client; assistant reply is persisted to SQLite after the stream completes.

### Layers

**API** (`backend/api/`)

- `app.py` – FastAPI app with an upload-size guard middleware (checks `Content-Length` *and* streams defensively).
- `documents.py` – `DocumentView` class-based view (via `fastapi-cbv`) on prefix `/v1/chats/{session_id}/documents`. Handles upload, list, and delete. `VectorStore` is injected via `Depends(VectorStore.from_settings)`.
- `messages.py` – `MessageView` CBV on prefix `/v1/chats/{session_id}/messages`. `POST /` retrieves context from Qdrant, streams the LLM reply as SSE (`event: chunk` / `event: done` / `event: error`), and persists both user and assistant messages to SQLite. `GET /` returns the ordered message history.
- `schemas.py` – Pydantic models: `IngestResponse`, `MessageRequest` (with `top_k`, `score_threshold`, `filters`), `MessageResponse`, `RetrievedChunk`, `MetadataFilter`, `StreamChunk`, `MessageHistoryItem`.

**Core** (`backend/core/`)

- `config.py` – `Settings` (Pydantic `BaseSettings`, singleton `settings`). Nested sub-settings: `IngestSettings`, `VectorStoreSettings`, `TextChunkSettings`, `SearchSettings`, `LLMSettings`. Uses `env_nested_delimiter="__"` so e.g. `VECTOR_STORE__COLLECTION_NAME` maps to `settings.vector_store.collection_name`. Groq and Qdrant API keys are validated against strict regex patterns.
- `ingest.py` – `DocumentIngestor` (Pydantic `BaseModel`). Binary-head validation via `verify_file_integrity` runs before Docling parses (guards against extension spoofing). Supported formats: `.pdf`, `.docx`, `.md`, `.xlsx`. Produces `Document` objects with `filename` and `page_number` in metadata.
- `chunking.py` – `TextChunker`. Uses `RecursiveCharacterTextSplitter` with a HuggingFace tokenizer (`BAAI/bge-large-en-v1.5` by default). Tokenizer and splitter are module-level singletons (thread-safe via `lru_cache` + `threading.Lock`).
- `vector_store.py` – `VectorStore` (Pydantic `BaseModel`). Wraps `QdrantVectorStore` (LangChain) with `QdrantClient` / `AsyncQdrantClient`. Supports local-path mode (sync-only fallback via `asyncio.to_thread`) and remote-URL mode (true async). Qdrant payload indexes on `metadata.session_id` (keyword) and `metadata.uploaded_at` (datetime) support scoped queries and TTL cleanup. Key methods: `ainit_collection`, `ainsert_docs`, `get_retriever`, `alist_documents`, `adelete_document`, `adelete_session`, `aclean_up_stale_vectors`.
- `database.py` – SQLite async engine via SQLAlchemy + SQLModel. `get_engine` and `get_session_factory` are `lru_cache` singletons. `init_db` / `close_db` are called from FastAPI lifespan. `get_session` is the FastAPI dependency yielding `AsyncSession`.
- `models.py` – SQLModel table models: `ChatSession` (PK: `id` str/UUID) and `ChatMessage` (PK: `id` int, FK to `chat_sessions.id`, `role` constrained to `'user'|'ai'`, `retrieved_chunks` stored as JSON).
- `llms/` – LLM client abstraction. `LLMClientProto` (Protocol) defines `astream_response(documents, question, chat_history)`. `LLMClientFactory.from_settings()` returns the configured client. Concrete implementations: `groq.py` (Groq), `openrouter.py` (OpenRouter). `prompt.py` holds the shared prompt template.

### Key Invariants

- `TextChunkSettings.chunk_overlap` must be strictly less than `chunk_size` (enforced by model validator).
- The SSE stream in `messages.py` uses a separate `async_session_factory()` context to persist the assistant reply after streaming completes — the request-scoped `db` session is already committed and cannot be reused inside the generator.
- `picture_classification` and `picture_description` toggles are silently forced off when `generate_picture_images=False`.
- File upload size is enforced twice: once via `Content-Length` header in middleware, once by streaming chunk accumulation in the route handler.

**Frontend** (`frontend/src/`)

- `types/api.ts` – TypeScript interfaces mirroring backend Pydantic schemas; keep in sync with `backend/api/schemas.py`.
- `lib/api.ts` – fetch/XHR helpers. File upload uses `XMLHttpRequest` (not `fetch`) to expose upload-progress events.
- `lib/sse.ts` – `streamMessage()`: `fetch` + `ReadableStream` SSE client. Uses `EventSource` cannot be used here because the endpoint is `POST`. Buffers incoming bytes, splits on `\n\n`, dispatches `onChunk` / `onDone` / `onError`. Pass an `AbortController.signal` to cancel.
- `context/SessionContext.tsx` – Session list is client-only (no backend list endpoint), persisted to `localStorage` under key `rag_sessions`. Active session ID comes from the URL param `/chat/:sessionId`.
- `hooks/useDocuments.ts` / `useMessages.ts` – TanStack Query wrappers. `useMessages` sets `staleTime: Infinity`; history is refreshed by calling `queryClient.invalidateQueries` after the SSE `done` event in `ChatContainer`.
- `components/chat/ChatContainer.tsx` – owns all streaming state (`streamingContent`, `isStreaming`, `streamingDone`, `streamingSources`). The `StreamingBubble` is rendered alongside the history list while streaming; after `done`, history is invalidated and the bubble resets.
- `app.py` (backend) – conditionally mounts `frontend/dist/` as static files at `/` if the directory exists. API routes registered before the mount take precedence.

### Key Frontend Invariants

- `EventSource` cannot be used for the SSE endpoint — it only supports GET. Always use the `fetch`+`ReadableStream` approach in `lib/sse.ts`.
- The `ChatInput` settings panel controls `top_k` (1–50, default 4) and an optional `score_threshold` (0.0–1.0). These are passed directly into the `MessageRequest` body.
- Session labels auto-update to the first 40 characters of the first question sent in that session.

## Testing

Tests are split into `tests/api/` and `tests/core/`, each with their own `conftest.py`.

**`tests/core/conftest.py`** fixtures:

- `reset_tokenizer_cache` (autouse) – clears `_get_cached_tokenizer` lru_cache, `TextChunker._recursive_text_splitter`, and `_INIT_LOCK` before each test to prevent state bleed.
- `mock_qdrant_clients` – patches `QdrantClient` and `AsyncQdrantClient` via `monkeypatch` at the `backend.core.vector_store` module level; returns `(mock_sync, mock_async)`.
- `vector_store` – parametrized over `local_path` and `remote_url`; uses `MockEmbeddings` (returns zero vectors at the configured `vector_size`) to avoid downloading HuggingFace models; sets `vs._vector_store` to a mocked `QdrantVectorStore`.
- `fast_ingest_config` (session-scoped) – `IngestSettings` with all heavy processing disabled (OCR, table structure, images).
- `db_engine` / `db_session` – in-memory SQLite via `sqlite+aiosqlite:///:memory:`, creates and drops all SQLModel tables around each test.

**`tests/api/conftest.py`** fixtures:

- `db_engine` / `db_session` – same in-memory SQLite pattern as above.
- `client` – `AsyncClient` (httpx ASGI transport) with `app.dependency_overrides` wiring `get_session` to the test DB session, `get_vector_store` to a mock, `_get_llm_client` to a mock async generator, and `get_session_factory` to a factory bound to the test engine (so the SSE background commit also hits the test DB). Overrides are cleared after each test.
- `mock_result` – a pre-built `UpdateResult(COMPLETED)` for Qdrant mutation assertions.
- `mock_pdf_bytes` – minimal PDF magic bytes for upload tests.

Coverage must stay ≥ 80% (`--cov-fail-under=80`). `asyncio_mode = "auto"` is set in `pyproject.toml` so async test functions need no decorator.

## Docker / Production

The `docker-compose.yml` runs four services: `nginx` (reverse proxy on port 80), `api` (FastAPI on port 8000, internal), `qdrant` (vector store, internal), and `postgres` (chat history DB, internal). In production the database backend switches from SQLite to PostgreSQL — set `DATABASE__URL=postgresql+asyncpg://...` in `.env.docker`.

- **Build and start**: `docker compose up --build`
- **Stop**: `docker compose down`
- HuggingFace and Docling model caches are persisted in named volumes (`hf_cache`, `docling_cache`) to avoid re-downloading on restart.
- The API health probe hits `GET /api/health`; nginx only starts after it passes.

## Environment Setup

Copy `.env.example` to `.env` (local dev) or `.env.docker` (Docker). Key variables:

| Variable                        | Default                                  | Notes                                                   |
| :-----------------------------: | :--------------------------------------: | :-----------------------------------------------------: |
| `DATABASE__URL`                 | `sqlite+aiosqlite:///./rag.db`           | Switch to `postgresql+asyncpg://...` in Docker          |
| `VECTOR_STORE__URL_OR_PATH`     | `./.qdrant_local/`                       | Set to `http://qdrant:6333/` in Docker                  |
| `VECTOR_STORE__EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must match `VECTOR_STORE__VECTOR_SIZE` (384 for MiniLM) |
| `LLM__PROVIDER`                 | `groq`                                   | Options: `groq`, `openrouter`                           |
| `LLM__API_KEY`                  | —                                        | Required                                                |

## Important Notes

- Pre‑commit enforces `black` formatting and `basedpyright` type checking; both must pass before a commit lands.
- `settings` is a module-level singleton instantiated at import time — use `monkeypatch.setattr` to override fields in tests rather than re-importing.
- When connecting to Qdrant Cloud (`*.qdrant.tech`), `VECTOR_STORE__API_KEY` is required and validated against a strict base64 pattern.
