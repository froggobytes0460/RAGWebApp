# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

- **Setup environment**: `uv sync`
- **Install pre‑commit hooks**: `uv run pre-commit install`
- **Run all tests**: `uv run pytest -q --tb=short --no-header`
- **Run a single test**: `uv run pytest path/to/test_file.py::test_name`
- **Run a single test file**: `uv run pytest tests/test_specific.py`
- **Run pre‑commit checks**: `uv run pre-commit run --all-files`
- **Lint / type‑check** (via pre‑commit): `uv run pre-commit run basedpyright --all-files`

## High‑Level Architecture

The repository consists of several key modules:

- **`backend/core/config.py`** – Central configuration via Pydantic BaseSettings.
- **`backend/core/ingest.py`** – Asynchronous document ingestion using Docling with binary‑head validation and optional OCR/table/image processing.
- **`backend/core/chunking.py`** – Logic for splitting extracted text into retrieval‑ready chunks.
- **`backend/core/vector_store.py`** – Async abstraction over vector‑store back‑ends (e.g., Qdrant, Chroma) handling upserts and similarity search.
- **`tests/`** – Pytest suite covering ingestion, chunking, vector‑store integration, and file‑integrity verification.

The project is an **asynchronous Python RAG web application** built around three logical layers:

1. **Configuration Layer** (`backend/core/config.py`)
   - Central `BaseSettings` singleton using Pydantic for all environment variables.
   - Provides typed access to API keys, vector store settings, and feature toggles.

2. **Ingestion Layer** (`backend/core/ingest.py`)
   - Uses IBM's **Docling** pipeline to extract text, tables, OCR, and image descriptions from many document formats.
   - Performs **binary‑head validation** to stop extension‑spoofing attacks before parsing.
   - Offers granular toggles (`enable_ocr`, `enable_tables`, `enable_image_desc`) to trade speed vs. fidelity.

3. **Vector Store / Retrieval Layer** (implemented in `backend/core/vectorstore/` – see the `feature/vectorstore` branch)
   - Abstracts storage back‑ends (e.g., Qdrant, Chroma) behind a common async interface.
   - Handles upserts, similarity search, and batch indexing.

### Supporting Packages

- **`tests/`** – pytest suite with fixtures for ingestion examples.
- **`.env.example`** – template for required environment variables.
- **`pyproject.toml` / `uv.lock`** – dependency management via **uv**.

## Development Workflow

1. **Initialize**: `uv sync`
2. **Create env file**: `cp .env.example .env` and fill in required keys.
3. **Install hooks**: `uv run pre-commit install`
4. **Run the test suite**: `uv run pytest`
5. **Iterate**: make changes → `uv run pre-commit run --all-files` → `uv run pytest`
6. **Commit**: hooks enforce formatting and type‑checking automatically.

## Important Notes

- Pre‑commit uses **basedpyright** for static type checking; failures must be resolved before committing.
- The codebase is a **prototype**; expect placeholder implementations and TODOs in feature branches.
- Vector‑store related code resides on the `feature/vectorstore` branch and may diverge from `main`.
