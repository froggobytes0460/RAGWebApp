# RAG Web Application

An async Retrieval-Augmented Generation (RAG) web application — upload documents, ask questions, get streaming answers grounded in your content.

> [!WARNING]
> This application is a prototype.

---

## How It Works

**Ingest:** Upload a document → background worker parses it (Docling) → chunks it → embeds and stores it in Qdrant. Progress streams back to the UI over SSE.

**Chat:** Send a message → retrieve relevant chunks from Qdrant → stream an LLM response back over SSE → persist the exchange to the database.

### Stack

| Layer | Technology |
| :---- | :--------- |
| Frontend | React 19, TypeScript, Vite, TailwindCSS 4, TanStack Query, React Router 7 |
| API | FastAPI — async, class-based views, slowapi rate limiting |
| Vector store | Qdrant with FastEmbed (`BAAI/bge-small-en-v1.5`) |
| Document parsing | IBM Docling — PDF, DOCX, XLSX, Markdown |
| Database | SQLite (dev) / PostgreSQL 17 (Docker) |
| LLM providers | Groq, OpenRouter — tenacity retry + SSE streaming |

---

## Quick Start

### Prerequisites

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- Node.js 18+

> Qdrant runs in local-path mode by default — no separate server required for development.

### 1. Install

```bash
git clone https://github.com/froggobytes0460/RAGWebApp.git
cd RAGWebApp
uv sync
uv run pre-commit install
cp .env.example .env
# Edit .env — at minimum set LLM__API_KEY
```

### 2. Run

```bash
# Terminal 1 — API (FastAPI dev server, logs to api.log)
NO_COLOR=1 uv run fastapi dev backend/api > api.log 2>&1 &

# Terminal 2 — Frontend (proxies /api → :8000)
cd frontend && npm install && npm run dev
```

Open **[localhost](http://localhost:5173)**.

### 3. Test

```bash
# Backend — 80% coverage threshold enforced
uv run pytest -q --tb=short --no-header

# Frontend
cd frontend && npm run test:coverage

# Lint + type-check
uv run pre-commit run --all-files
```

---

## Docker (Production)

```bash
docker compose up --build   # nginx + api + qdrant + postgres
docker compose down
```

Services: `nginx` (port 80) → `api` (:8000) + `qdrant` + `postgres:17`. Model caches (`hf_cache`, `docling_cache`) are persisted in named volumes.

Add to `.env.docker` to switch from SQLite/local Qdrant to the containerised backends:

```env
DATABASE__URI=postgresql+asyncpg://user:password@postgres:5432/ragdb
VECTOR_STORE__URL_OR_PATH=http://qdrant:6333/
```

---

## Environment Variables

Full reference in [`.env.example`](./.env.example) and [`backend/core/config.py`](./backend/core/config.py).

| Variable | Default | Notes |
| :------- | :------ | :---- |
| `LLM__PROVIDER` | `groq` | `groq` or `openrouter` |
| `LLM__MODEL_NAME` | `llama-3.3-70b-versatile` | Provider-specific model name |
| `LLM__API_KEY` | — | **Required** |
| `DATABASE__URI` | `sqlite+aiosqlite:///./rag.db` | Switch to `postgresql+asyncpg://...` in Docker |
| `VECTOR_STORE__URL_OR_PATH` | `./.qdrant_local/` | Switch to `http://qdrant:6333/` in Docker |
| `VECTOR_STORE__EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Must match `VECTOR_STORE__VECTOR_SIZE` |
| `VECTOR_STORE__VECTOR_SIZE` | `384` | Embedding dimensions |
| `VECTOR_STORE__TTL` | `86400` | Stale-vector TTL (seconds) |
| `TEXT_CHUNK__CHUNK_SIZE` | `500` | Max tokens per chunk |
| `TEXT_CHUNK__CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `INGEST__MAX_FILE_SIZE` | `50` | Upload size limit (MB) |

---

## API Reference

| Method | Path | Description | Rate limit |
| :----- | :--- | :---------- | :--------: |
| `POST` | `/v1/chats/{session_id}/documents` | Upload and ingest a document | 10 / min |
| `GET` | `/v1/chats/{session_id}/documents` | List documents for a session | — |
| `GET` | `/v1/chats/{session_id}/documents/jobs/{job_id}/progress` | SSE stream — ingestion progress | — |
| `DELETE` | `/v1/chats/{session_id}/documents/{filename}` | Delete a document | — |
| `POST` | `/v1/chats/{session_id}/messages` | Send a message; stream LLM reply as SSE | 20 / min |
| `GET` | `/v1/chats/{session_id}/messages` | Retrieve message history | — |
| `GET` | `/api/health` | Dependency health check — returns `{"status", "version", "dependencies": {"database", "vector_store"}}` each with `status` and `latency_ms`; HTTP 200 if all healthy, 503 if any degraded | — |

---

## Repository Structure

```text
├── backend/
│   ├── api/            # FastAPI routes, schemas, rate limiter, app state
│   └── core/
│       ├── llms/       # Groq + OpenRouter clients, RAG prompt template
│       ├── config.py   # Pydantic BaseSettings singleton
│       ├── ingest.py   # Docling loader + binary-head validation
│       ├── chunking.py # Tokenizer-aware text splitter (LRU-cached)
│       ├── vector_store.py     # Qdrant wrapper (FastEmbed)
│       ├── ingestion_worker.py # Background async ingest worker
│       ├── database.py # Async SQLAlchemy engine + session factory
│       └── models.py   # SQLModel tables (ChatSession, ChatMessage, IngestionJob)
├── frontend/src/       # React SPA — components, hooks, SSE client, TanStack Query
├── tests/
│   ├── api/            # Integration tests (httpx AsyncClient)
│   └── core/           # Unit tests — ingest, chunking, vector store, LLM clients
├── nginx/              # Reverse-proxy config
├── .github/workflows/  # CI: backend, frontend, docker
├── Dockerfile          # Multi-stage build
└── docker-compose.yml  # Full production stack
```

---

## CI

| Workflow | Trigger | Gates |
| :------- | :------ | :---- |
| `backend.yaml` | Changes to `backend/`, `tests/`, `pyproject.toml` | black → basedpyright → pytest ≥ 80% coverage |
| `frontend.yaml` | Changes to `frontend/` | ESLint → vitest coverage → tsc + Vite build |
| `docker.yaml` | Changes to `Dockerfile`, `docker-compose.yml`, `backend/`, `frontend/` | `docker compose build` |

---

## Security

- **Extension spoofing protection** — binary-head validation runs before Docling parses any file.
- **Double upload-size enforcement** — `Content-Length` header check in middleware, plus streaming chunk accumulation in the route handler.
- **API key validation** — Groq and Qdrant Cloud keys validated against strict regex patterns at startup.
- **Rate limiting** — 10 req/min on document upload, 20 req/min on chat.
