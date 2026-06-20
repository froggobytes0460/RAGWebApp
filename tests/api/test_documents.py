# pyright: reportAny=none

from collections.abc import AsyncGenerator
from io import BytesIO

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_mock
from qdrant_client.models import UpdateResult
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.api.app import app
from backend.api.documents import get_vector_store
from backend.core.database import get_session


class TestCreateDocument:
    async def test_returns_202_on_valid_upload(
        self,
        client: AsyncClient,
        mock_pdf_bytes: bytes,
        db_engine: AsyncEngine,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        _ = mocker.patch(
            "backend.api.documents.get_session_factory",
            return_value=async_sessionmaker(
                bind=db_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            ),
        )

        resp = await client.post(
            url="/api/v1/chats/sess-abc/documents/",
            files={
                "file": (
                    "report.pdf",
                    BytesIO(initial_bytes=mock_pdf_bytes),
                    "application/pdf",
                )
            },
        )

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    async def test_400_on_unsupported_extension(
        self,
        client: AsyncClient,
    ) -> None:
        resp = await client.post(
            url="/api/v1/chats/sess-abc/documents/",
            files={"file": ("image.png", BytesIO(b"PNG data"), "image/png")},
        )

        assert resp.status_code == 400
        assert "Unsupported file format" in resp.json()["detail"]

    async def test_413_when_file_exceeds_size_limit(
        self,
        client: AsyncClient,
        mock_pdf_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend.core.config import settings

        monkeypatch.setattr(settings.ingest, "max_file_size", 0)

        resp = await client.post(
            url="/api/v1/chats/sess-abc/documents/",
            files={
                "file": (
                    "big.pdf",
                    BytesIO(initial_bytes=mock_pdf_bytes * 10),
                    "application/pdf",
                )
            },
        )

        assert resp.status_code == 413


class TestListDocuments:
    async def test_returns_200_with_document_list(
        self,
        db_session: AsyncSession,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        docs = [
            {"filename": "a.pdf", "uploaded_at": "2024-01-01T00:00:00"},
            {"filename": "b.pdf", "uploaded_at": "2024-01-02T00:00:00"},
        ]
        mock_vs = mocker.MagicMock()
        mock_vs.alist_documents = mocker.AsyncMock(return_value=docs)

        async def _override_get_session() -> AsyncGenerator[AsyncSession]:
            yield db_session

        app.dependency_overrides[get_session] = _override_get_session
        app.dependency_overrides[get_vector_store] = lambda: mock_vs

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            resp = await http_client.get("/api/v1/chats/sess-list/documents/")

        app.dependency_overrides.clear()

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["filename"] == "a.pdf"

    async def test_returns_empty_list_for_empty_session(
        self,
        db_session: AsyncSession,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_vs = mocker.MagicMock()
        mock_vs.alist_documents = mocker.AsyncMock(return_value=[])

        async def _override_get_session() -> AsyncGenerator[AsyncSession]:
            yield db_session

        app.dependency_overrides[get_session] = _override_get_session
        app.dependency_overrides[get_vector_store] = lambda: mock_vs

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            resp = await http_client.get("/api/v1/chats/sess-empty/documents/")

        app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert resp.json() == []


class TestDeleteDocument:
    async def test_returns_202_on_successful_delete(
        self,
        client: AsyncClient,
        mocker: pytest_mock.MockerFixture,
        mock_result: UpdateResult,
    ) -> None:
        mock_vs = app.dependency_overrides[get_vector_store]()
        mock_vs.adelete_document = mocker.AsyncMock(return_value=mock_result)

        resp = await client.delete("/api/v1/chats/sess-del/documents/report.pdf")

        assert resp.status_code == 202
        assert resp.json()["status"] == "completed"

    async def test_delete_calls_vector_store_with_correct_args(
        self,
        db_session: AsyncSession,
        mock_result: UpdateResult,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_vs = mocker.MagicMock()
        mock_vs.adelete_document = mocker.AsyncMock(return_value=mock_result)

        async def _override_get_session() -> AsyncGenerator[AsyncSession]:
            yield db_session

        app.dependency_overrides[get_session] = _override_get_session
        app.dependency_overrides[get_vector_store] = lambda: mock_vs

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            resp = await http_client.delete(
                "/api/v1/chats/sess-del/documents/my_report.pdf"
            )

        app.dependency_overrides.clear()

        assert resp.status_code == 202
        mock_vs.adelete_document.assert_awaited_once_with(
            session_id="sess-del", filename="my_report.pdf"
        )

    @pytest.mark.parametrize(
        argnames="filename",
        argvalues=["doc with spaces.pdf", "résumé.pdf", "report-2024.docx"],
        ids=["spaces", "unicode", "hyphenated"],
    )
    async def test_delete_accepts_various_filenames(
        self,
        filename: str,
        db_session: AsyncSession,
        mock_result: UpdateResult,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_vs = mocker.MagicMock()
        mock_vs.adelete_document = mocker.AsyncMock(return_value=mock_result)

        async def _override_get_session() -> AsyncGenerator[AsyncSession]:
            yield db_session

        app.dependency_overrides[get_session] = _override_get_session
        app.dependency_overrides[get_vector_store] = lambda: mock_vs

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            resp = await http_client.delete(
                url=f"/api/v1/chats/sess-del/documents/{filename}"
            )

        app.dependency_overrides.clear()
        assert resp.status_code == 202
