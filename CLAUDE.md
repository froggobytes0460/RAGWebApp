# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

- `uv sync`: Initialize or update Python environment
- `uv run pytest -q --tb=short --no-header`: Run all unit tests
- `uv run pre-commit install`: Install pre-commit hooks
- `uv run pre-commit run --all-files`: Run pre-commit hooks manually

## High-Level Code Architecture

This is a Python-based asynchronous RAG (Retrieval-Augmented Generation) web application with the following core components:

### Backend Core (`backend/core/`)

- **config.py**: Global Pydantic BaseSettings singleton for configuration management
- **ingest.py**: Asynchronous Docling loader with file integrity validation and multi-format document extraction

### Key Features

- **Asynchronous Multi-Format Ingestor**: Non-blocking document extraction using IBM's Docling pipeline
- **Cryptographic Sandbox Protection**: Binary head validation to prevent extension spoofing attacks
- **Granular Layout Toggles**: Configurable processing for OCR, table parsing, and image descriptions

### Testing (`tests/`)

- **conftest.py**: Test configuration
- **test_ingest.py**: Unit tests for ingestion functionality
- **fixtures/ingest/**: Test data files for ingestion tests

### Environment Configuration

- `.env.example`: Template for environment variables
- Configuration logic in `backend/core/config.py` defines all available settings

## Development Workflow

1. Initialize environment: `uv sync`
2. Set up environment: `cp .env.example .env`
3. Install pre-commit hooks: `uv run pre-commit install`
4. Run tests: `uv run pytest`
5. Code changes are automatically validated on commit via pre-commit hooks

## Important Notes

- Pre-commit hooks enforce code style and linting using `basedpyright`
- The application is a prototype for a production RAG system
- Document processing speed vs. fidelity can be balanced through configuration
