"""Test utilities for RAGWebApp.
Provides a helper to create a corrupted temporary file for integrity verification tests.
"""

import threading

import pytest

from backend.core import chunking
from backend.core.chunking import (
    TextChunker,
    _get_cached_tokenizer,  # pyright: ignore[reportPrivateUsage]
)
from backend.core.config import IngestSettings


@pytest.fixture(scope="session")
def fast_ingest_config() -> IngestSettings:
    return IngestSettings(
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=False,
        generate_picture_images=False,
        do_picture_classification=False,
        do_picture_description=False,
    )


@pytest.fixture(autouse=True)
def reset_tokenizer_cache(monkeypatch: pytest.MonkeyPatch):
    """Clear the lru_cache and reset class variables before each test."""
    _get_cached_tokenizer.cache_clear()
    TextChunker._recursive_text_splitter = None  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(chunking, "_INIT_LOCK", threading.Lock())
    yield
