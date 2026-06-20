# pyright: reportAny=none

from io import BytesIO

from httpx import AsyncClient
import pytest
import pytest_mock
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.api.limiter import limiter


@pytest.fixture(autouse=True)
def reset_limiter() -> None:
    limiter.reset()


class TestDocumentRateLimit:
    async def test_eleventh_upload_returns_429(
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
            assert resp.status_code == 202

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
            url="/api/v1/chats/sess1/documents/",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(initial_bytes=mock_pdf_bytes),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 202


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
