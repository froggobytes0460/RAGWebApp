# pyright: reportAny=none

from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_mock
from qdrant_client.http.models import CollectionsResponse

from backend.api import app
from backend.api.state import AppState


@pytest.fixture
async def health_client() -> AsyncGenerator[AsyncClient]:
    app.typed_state = AppState()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def test_health_ok(
    health_client: AsyncClient, mocker: pytest_mock.MockerFixture
) -> None:
    mock_conn = mocker.AsyncMock()
    mock_conn.__aenter__ = mocker.AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_conn.exec_driver_sql = mocker.AsyncMock()

    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value = mock_conn

    mock_vs = mocker.MagicMock()
    mock_vs.async_client = mocker.AsyncMock()
    mock_vs.async_client.get_collections = mocker.AsyncMock(
        return_value=CollectionsResponse(collections=[])
    )

    with (
        mocker.patch("backend.api.app.get_engine", return_value=mock_engine),
        mocker.patch("backend.api.app.get_vector_store", return_value=mock_vs),
    ):
        resp = await health_client.get(url="/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["dependencies"]["database"]["status"] == "ok"
    assert body["dependencies"]["vector_store"]["status"] == "ok"
    assert body["dependencies"]["database"]["latency_ms"] is not None


async def test_health_db_degraded(
    health_client: AsyncClient, mocker: pytest_mock.MockerFixture
) -> None:
    mock_conn = mocker.AsyncMock()
    mock_conn.__aenter__ = mocker.AsyncMock(
        side_effect=Exception("DB connection refused")
    )
    mock_conn.__aexit__ = mocker.AsyncMock(return_value=False)

    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value = mock_conn

    mock_vs = mocker.MagicMock()
    mock_vs.async_client = mocker.AsyncMock()
    mock_vs.async_client.get_collections = mocker.AsyncMock(
        return_value=CollectionsResponse(collections=[])
    )

    with (
        mocker.patch("backend.api.app.get_engine", return_value=mock_engine),
        mocker.patch("backend.api.app.get_vector_store", return_value=mock_vs),
    ):
        resp = await health_client.get(url="/api/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["database"]["status"] == "degraded"
    assert "DB connection refused" in body["dependencies"]["database"]["detail"]
    assert body["dependencies"]["vector_store"]["status"] == "ok"


async def test_health_vector_store_degraded(
    health_client: AsyncClient, mocker: pytest_mock.MockerFixture
) -> None:
    mock_conn = mocker.AsyncMock()
    mock_conn.__aenter__ = mocker.AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_conn.exec_driver_sql = mocker.AsyncMock()

    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value = mock_conn

    mock_vs = mocker.MagicMock()
    mock_vs.async_client = mocker.AsyncMock()
    mock_vs.async_client.get_collections = mocker.AsyncMock(
        side_effect=Exception("Qdrant unreachable")
    )

    with (
        mocker.patch("backend.api.app.get_engine", return_value=mock_engine),
        mocker.patch("backend.api.app.get_vector_store", return_value=mock_vs),
    ):
        resp = await health_client.get("/api/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["vector_store"]["status"] == "degraded"
    assert "Qdrant unreachable" in body["dependencies"]["vector_store"]["detail"]


async def test_health_sync_client_fallback(
    health_client: AsyncClient, mocker: pytest_mock.MockerFixture
) -> None:
    """Covers the local-path mode where async_client is None."""
    mock_conn = mocker.AsyncMock()
    mock_conn.__aenter__ = mocker.AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_conn.exec_driver_sql = mocker.AsyncMock()

    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value = mock_conn

    mock_vs = mocker.MagicMock()
    mock_vs.async_client = None
    mock_vs.client = mocker.MagicMock()
    mock_vs.client.get_collections = mocker.MagicMock(
        return_value=CollectionsResponse(collections=[])
    )

    with (
        mocker.patch("backend.api.app.get_engine", return_value=mock_engine),
        mocker.patch("backend.api.app.get_vector_store", return_value=mock_vs),
    ):
        resp = await health_client.get(url="/api/health")

    assert resp.status_code == 200
    assert resp.json()["dependencies"]["vector_store"]["status"] == "ok"
