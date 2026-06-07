# RAG Web-Application

A high-performance, asynchronously optimized Retrieval-Augmented Generation (RAG) web application.

> [!WARNING]
> **Disclaimer:** This application is an prototype of the real RAG webapp.

## Environment Variables

Environment variable example is given in this [example .env file](./.env.example). Also check the [configuration logic](./backend/core/config.py) for more info.

## Key Architectural Features

* **Asynchronous Multi-Format Ingestor**: Non-blocking document extraction layer powered by IBM's Docling pipeline.
* **Cryptographic Sandbox Protection**: Active binary head validation intercepts extension spoofing attacks before document parsing runs.
* **Granular Layout Toggles**: Configurable heavy document processing (OCR, table parsing, image descriptions) to balance extraction speed against fidelity.

---

## Core Repository Topology

```text
├── backend/
│   └── core/
│       ├── config.py       # Global Pydantic BaseSettings singleton orchestrator
│       └── ingest.py       # Asynchronous Docling loader and file integrity layer
├── tests/                  # Unit testing
│   ├── conftest.py
│   ├── fixtures/
│   │   └── ingest/
│   └── test_ingest.py
├── .env.example            # Environment variables blueprint mapping
└── README.md               # System specification and documentation
```

### Development Quality Controls (Pre-commit Hooks)

This repository includes a `.pre-commit-config.yaml` file to enforce uniform code styling, linting, and formatting safeguards before changes are committed.

```bash
# Step 1: Install the git hook scripts inside your local .git directory
uv run pre-commit install

# Step 2: (Optional) Run the hooks manually against all files to ensure baseline compliance
uv run pre-commit run --all-files
```

Once installed, these validations run automatically every time you execute `git commit`. If a hook fails (e.g., formatting issues with `basedpyright`), fix the errors and re-stage the files before committing again.

## Installation & Environment Initialization

### 1. Local Environment

Clone the github repo and set up the environment variables.

```bash
# Step 1: Clone the repository (replace with the actual URL)
git clone https://github.com/froggobytes0460/RAGWebApp.git rag-web-application/
cd rag-web-application

# Step 2: Initialize `uv` environment
uv sync

# Step 3: Copy the example environment file to create your active configuration
cp .env.example .env
```

> [!WARNING]
> **Disclaimer:** Incomplete, will be added once developement of app finishes.

---

## Running the Test Suite

Execute the unit tests using `uv` to ensure your environment is configured correctly:

```bash
# Run all tests using pytest via uv
uv run pytest
```

---
