# pyright: reportExplicitAny=none
# pyright: reportUnknownLambdaType=none

from collections.abc import AsyncIterator
from typing import Any, Never
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, HumanMessage
from openrouter import errors as openrouter_errors
from pydantic import SecretStr
import pytest
from pytest_mock import MockerFixture

from backend.core.config import settings
from backend.core.llms.openrouter import LLMOpenRouterClient


def _make_chunk(content: str) -> AIMessageChunk:
    return AIMessageChunk(content=content)


async def _async_gen(*items: AIMessageChunk) -> AsyncIterator[AIMessageChunk]:
    for item in items:
        yield item


def _mock_openrouter_client(mocker: MockerFixture) -> LLMOpenRouterClient:
    """Construct an LLMOpenRouterClient bypassing Pydantic validation."""
    return LLMOpenRouterClient.model_construct(openrouter_client=mocker.MagicMock())


def _capture_chat_openrouter_kwargs(mocker: MockerFixture) -> dict[str, object]:
    """Invoke LLMOpenRouterClient.from_settings and return kwargs passed to ChatOpenRouter.

    Patches ChatOpenRouter to capture args, then patches Pydantic __init__ to bypass
    validation so cls(openrouter_client=<mock>) succeeds without type errors.
    """
    captured: dict[str, object] = {}

    def fake_chat_openrouter(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return mocker.MagicMock()

    _ = mocker.patch(
        "backend.core.llms.openrouter.ChatOpenRouter", side_effect=fake_chat_openrouter
    )

    original_init = LLMOpenRouterClient.__init__
    _ = mocker.patch.object(
        target=LLMOpenRouterClient, attribute="__init__", return_value=None
    )

    try:
        _ = LLMOpenRouterClient.from_settings()
    finally:
        LLMOpenRouterClient.__init__ = original_init  # type: ignore[method-assign]

    return captured


class TestLLMOpenRouterClientFromSettings:
    def test_passes_all_settings_to_chat_openrouter(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings.llm, "model_name", "mistral-7b", raising=False)
        monkeypatch.setattr(settings.llm, "temperature", 0.2, raising=False)
        monkeypatch.setattr(settings.llm, "max_output_token", 512, raising=False)
        monkeypatch.setattr(
            settings.llm, "api_key", SecretStr("test-key"), raising=False
        )

        kwargs = _capture_chat_openrouter_kwargs(mocker)

        assert kwargs["model"] == "mistral-7b"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["max_retries"] == 0

    def test_passes_model_from_settings(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings.llm, "model_name", "gpt-4o", raising=False)

        kwargs = _capture_chat_openrouter_kwargs(mocker)

        assert kwargs["model"] == "gpt-4o"

    def test_max_retries_disabled_on_client(self, mocker: MockerFixture) -> None:
        kwargs = _capture_chat_openrouter_kwargs(mocker)

        assert kwargs["max_retries"] == 0


class TestLLMOpenRouterClientRetryableErrors:
    def test_includes_too_many_requests_error(self) -> None:
        assert (
            openrouter_errors.TooManyRequestsResponseError
            in LLMOpenRouterClient.retryable_openrouter_errors
        )

    def test_includes_request_timeout_error(self) -> None:
        assert (
            openrouter_errors.RequestTimeoutResponseError
            in LLMOpenRouterClient.retryable_openrouter_errors
        )

    def test_includes_bad_gateway_error(self) -> None:
        assert (
            openrouter_errors.BadGatewayResponseError
            in LLMOpenRouterClient.retryable_openrouter_errors
        )

    def test_includes_service_unavailable_error(self) -> None:
        assert (
            openrouter_errors.ServiceUnavailableResponseError
            in LLMOpenRouterClient.retryable_openrouter_errors
        )


class TestLLMOpenRouterClientAstreamResponse:
    async def test_yields_content_from_chunks(self, mocker: MockerFixture) -> None:
        chunks = [_make_chunk(content="foo"), _make_chunk(content="bar")]
        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = lambda _input: _async_gen(*chunks)

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        results = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]
        assert results == ["foo", "bar"]

    async def test_skips_empty_content_chunks(self, mocker: MockerFixture) -> None:
        chunks = [_make_chunk(""), _make_chunk("real"), _make_chunk("")]
        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = lambda _input: _async_gen(*chunks)

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        results = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]
        assert results == ["real"]

    async def test_concatenates_document_context(self, mocker: MockerFixture) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk("ok")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = fake_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        docs = [Document(page_content="Alpha"), Document(page_content="Beta")]
        _ = [c async for c in client.astream_response(documents=docs, question="q")]
        assert captured["context"] == "Alpha\n\nBeta"

    async def test_passes_question_in_payload(self, mocker: MockerFixture) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk(content="ans")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = fake_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        _ = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="specific question"
            )
        ]
        assert captured["question"] == "specific question"

    async def test_defaults_chat_history_to_empty_list(
        self, mocker: MockerFixture
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk("ok")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = fake_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        _ = [
            c
            async for c in client.astream_response([Document(page_content="ctx")], "q")
        ]
        assert captured["chat_history"] == []

    async def test_passes_chat_history_when_provided(
        self, mocker: MockerFixture
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk("ok")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = fake_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        history = [HumanMessage(content="prior")]
        _ = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")],
                question="q",
                chat_history=history,
            )
        ]
        assert captured["chat_history"] == history

    async def test_handles_empty_document_list(self, mocker: MockerFixture) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk(content="ok")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = fake_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )

        _ = [c async for c in client.astream_response(documents=[], question="q")]
        assert captured["context"] == ""

    async def test_retryable_error_is_retried(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        async def flaky_astream(_: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise openrouter_errors.TooManyRequestsResponseError(
                    data=MagicMock(), raw_response=MagicMock()
                )
            yield _make_chunk(content="ok")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = flaky_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )
        _ = mocker.patch(
            "backend.core.llms.openrouter.wait_exponential_jitter",
            return_value=lambda _: 0,
        )
        monkeypatch.setattr(settings.llm, "max_retries", 3, raising=False)

        results = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]

        assert call_count == 2
        assert "ok" in results

    async def test_non_retryable_error_propagates(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def bad_astream(_: dict[str, Any]) -> Never:
            raise RuntimeError("boom")
            yield _make_chunk(content="never")  # pyright: ignore[reportUnreachable]

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = bad_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )
        monkeypatch.setattr(settings.llm, "max_retries", 1, raising=False)

        with pytest.raises(RuntimeError, match="boom"):
            _ = [
                c
                async for c in client.astream_response(
                    documents=[Document(page_content="ctx")], question="q"
                )
            ]

    async def test_mid_stream_retryable_error_yields_interrupted_message(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = openrouter_errors.BadGatewayResponseError(MagicMock(), MagicMock())
        first_call = True

        async def fake_astream(_: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            nonlocal first_call
            if first_call:
                first_call = False
                yield _make_chunk(content="partial")
                raise error
            yield _make_chunk(content="ok")

        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = fake_astream

        client = _mock_openrouter_client(mocker)
        _ = mocker.patch.object(
            target=type(client),
            attribute="runnable_lcel",
            new_callable=lambda: property(
                fget=lambda self: mock_runnable  # pyright: ignore[reportAny]
            ),
        )
        monkeypatch.setattr(settings.llm, "max_retries", 1, raising=False)

        results = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]
        assert any("Stream interrupted" in r for r in results)
