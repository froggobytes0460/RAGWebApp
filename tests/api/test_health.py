# pyright: reportAny=none

from unittest.mock import MagicMock

from httpx import AsyncClient
import pytest
import pytest_mock
from qdrant_client.http.models import CollectionsResponse


@pytest.fixture
def mock_db_engine(mocker: pytest_mock.MockerFixture) -> MagicMock:
    mock_conn = mocker.AsyncMock()
    mock_conn.__aenter__ = mocker.AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_conn.exec_driver_sql = mocker.AsyncMock()
    engine = mocker.MagicMock()
    engine.connect.return_value = mock_conn
    return engine


@pytest.fixture
def mock_async_vs(mocker: pytest_mock.MockerFixture) -> MagicMock:
    vs = mocker.MagicMock()
    vs._client = mocker.AsyncMock()
    vs._client.get_collections = mocker.AsyncMock(
        return_value=CollectionsResponse(collections=[])
    )
    return vs


async def test_health_liveness(client: AsyncClient) -> None:
    resp = await client.get(url="/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "dependencies" not in body


async def test_deep_health_ok(
    client: AsyncClient,
    mocker: pytest_mock.MockerFixture,
    mock_db_engine: MagicMock,
    mock_async_vs: MagicMock,
) -> None:
    _ = mocker.patch("backend.api.app.get_engine", return_value=mock_db_engine)
    _ = mocker.patch("backend.api.app._get_vector_store", return_value=mock_async_vs)
    resp = await client.get(url="/api/health/deep")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["dependencies"]["database"]["status"] == "ok"
    assert body["dependencies"]["vector_store"]["status"] == "ok"
    assert body["dependencies"]["database"]["latency_ms"] is not None


async def test_deep_health_db_degraded(
    client: AsyncClient,
    mocker: pytest_mock.MockerFixture,
    mock_async_vs: MagicMock,
) -> None:
    mock_conn = mocker.AsyncMock()
    mock_conn.__aenter__ = mocker.AsyncMock(
        side_effect=Exception("DB connection refused")
    )
    mock_conn.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value = mock_conn

    _ = mocker.patch("backend.api.app.get_engine", return_value=mock_engine)
    _ = mocker.patch("backend.api.app._get_vector_store", return_value=mock_async_vs)
    resp = await client.get(url="/api/health/deep")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["database"]["status"] == "degraded"
    assert "DB connection refused" in body["dependencies"]["database"]["detail"]
    assert body["dependencies"]["vector_store"]["status"] == "ok"


async def test_deep_health_vector_store_degraded(
    client: AsyncClient,
    mocker: pytest_mock.MockerFixture,
    mock_db_engine: MagicMock,
) -> None:
    mock_vs = mocker.MagicMock()
    mock_vs._client = mocker.AsyncMock()
    mock_vs._client.get_collections = mocker.AsyncMock(
        side_effect=Exception("Qdrant unreachable")
    )

    _ = mocker.patch("backend.api.app.get_engine", return_value=mock_db_engine)
    _ = mocker.patch("backend.api.app._get_vector_store", return_value=mock_vs)
    resp = await client.get("/api/health/deep")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["vector_store"]["status"] == "degraded"
    assert "Qdrant unreachable" in body["dependencies"]["vector_store"]["detail"]
