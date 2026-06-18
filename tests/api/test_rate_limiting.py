# pyright: reportAny=none

from io import BytesIO

from httpx import AsyncClient
from langchain_core.documents import Document
import pytest
import pytest_mock

from backend.api.app import app
from backend.api.documents import get_vector_store
from backend.api.limiter import limiter


@pytest.fixture(autouse=True)
def reset_limiter() -> None:
    limiter.reset()


class TestDocumentRateLimit:
    async def test_eleventh_upload_returns_429(
        self,
        client: AsyncClient,
        mock_pdf_bytes: bytes,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        parsed_doc = Document(
            page_content="hello",
            metadata={"filename": "test.pdf", "page_number": 1},
        )
        _ = mocker.patch(
            "backend.api.documents.DocumentIngestor.ingest_async",
            new=mocker.AsyncMock(return_value=[parsed_doc]),
        )
        _ = mocker.patch(
            "backend.api.documents.TextChunker.achunk_text",
            new=mocker.AsyncMock(return_value=[parsed_doc]),
        )
        mock_vs = app.dependency_overrides[get_vector_store]()
        mock_vs.ainsert_docs = mocker.AsyncMock(return_value=None)

        for _ in range(10):
            resp = await client.post(
                url="/api/v1/chats/sess1/documents/",
                files={
                    "file": (
                        "test.pdf",
                        BytesIO(initial_bytes=mock_pdf_bytes),
                        "application/pdf",
                    )
                },
            )
            assert resp.status_code == 201

        resp = await client.post(
            url="/api/v1/chats/sess1/documents/",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(initial_bytes=mock_pdf_bytes),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 429

    async def test_rate_limit_resets_between_test_runs(
        self,
        client: AsyncClient,
        mock_pdf_bytes: bytes,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        parsed_doc = Document(
            page_content="hello",
            metadata={"filename": "test.pdf", "page_number": 1},
        )
        _ = mocker.patch(
            "backend.api.documents.DocumentIngestor.ingest_async",
            new=mocker.AsyncMock(return_value=[parsed_doc]),
        )
        _ = mocker.patch(
            "backend.api.documents.TextChunker.achunk_text",
            new=mocker.AsyncMock(return_value=[parsed_doc]),
        )
        mock_vs = app.dependency_overrides[get_vector_store]()
        mock_vs.ainsert_docs = mocker.AsyncMock(return_value=None)

        resp = await client.post(
            url="/api/v1/chats/sess1/documents/",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(initial_bytes=mock_pdf_bytes),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 201


class TestMessageRateLimit:
    async def test_twenty_first_message_returns_429(
        self,
        client: AsyncClient,
    ) -> None:
        for _ in range(20):
            resp = await client.post(
                url="/api/v1/chats/sess1/messages/",
                json={"question": "hello"},
            )
            assert resp.status_code == 201

        resp = await client.post(
            url="/api/v1/chats/sess1/messages/",
            json={"question": "hello"},
        )
        assert resp.status_code == 429

    async def test_rate_limit_resets_between_test_runs(
        self,
        client: AsyncClient,
    ) -> None:
        resp = await client.post(
            url="/api/v1/chats/sess1/messages/",
            json={"question": "hello"},
        )
        assert resp.status_code == 201

    async def test_429_body_contains_error_detail(
        self,
        client: AsyncClient,
    ) -> None:
        for _ in range(20):
            _ = await client.post(
                url="/api/v1/chats/sess1/messages/",
                json={"question": "hello"},
            )

        resp = await client.post(
            url="/api/v1/chats/sess1/messages/",
            json={"question": "hello"},
        )
        assert resp.status_code == 429
        assert "error" in resp.json()
