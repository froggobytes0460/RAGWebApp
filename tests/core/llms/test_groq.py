# pyright: reportExplicitAny=none
# pyright: reportUnknownLambdaType=none

from collections.abc import AsyncIterator, Callable
from typing import Any, Never
from unittest.mock import MagicMock

import groq
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import SecretStr
import pytest
from pytest_mock import MockerFixture

from backend.core.config import settings
from backend.core.llms.groq import LLMGroqClient


def _make_chunk(content: str) -> AIMessageChunk:
    return AIMessageChunk(content=content)


async def _async_gen(*items: AIMessageChunk) -> AsyncIterator[AIMessageChunk]:
    for item in items:
        yield item


def _mock_groq_client(mocker: MockerFixture) -> LLMGroqClient:
    """Construct an LLMGroqClient bypassing Pydantic validation."""
    return LLMGroqClient.model_construct(groq_client=mocker.MagicMock())


def _capture_chat_groq_kwargs(mocker: MockerFixture) -> dict[str, object]:
    """Invoke LLMGroqClient.from_settings and return the kwargs passed to ChatGroq constructor.

    Strategy: patch ChatGroq to capture args, then patch the Pydantic model's __init__ to
    bypass validation so cls(groq_client=<mock>) succeeds without type errors.
    """
    captured: dict[str, object] = {}

    def fake_chat_groq(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return mocker.MagicMock()

    _ = mocker.patch("backend.core.llms.groq.ChatGroq", side_effect=fake_chat_groq)

    original_init = LLMGroqClient.__init__
    _ = mocker.patch.object(
        target=LLMGroqClient, attribute="__init__", return_value=None
    )

    try:
        _ = LLMGroqClient.from_settings()
    finally:
        LLMGroqClient.__init__ = original_init

    return captured


class TestLLMGroqClientFromSettings:
    def test_passes_all_settings_to_chat_groq(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings.llm, "model_name", "llama3-8b-8192", raising=False)
        monkeypatch.setattr(settings.llm, "temperature", 0.2, raising=False)
        monkeypatch.setattr(settings.llm, "max_output_token", 512, raising=False)
        monkeypatch.setattr(
            settings.llm, "api_key", SecretStr("test-key"), raising=False
        )

        kwargs = _capture_chat_groq_kwargs(mocker)

        assert kwargs["model"] == "llama3-8b-8192"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["max_retries"] == 0

    def test_passes_temperature_from_settings(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings.llm, "temperature", 0.9, raising=False)

        kwargs = _capture_chat_groq_kwargs(mocker)

        assert kwargs["temperature"] == 0.9

    def test_max_retries_disabled_on_client(self, mocker: MockerFixture) -> None:
        kwargs = _capture_chat_groq_kwargs(mocker)

        assert kwargs["max_retries"] == 0


class TestLLMGroqClientRetryableErrors:
    def test_includes_rate_limit_error(self) -> None:
        assert groq.RateLimitError in LLMGroqClient.retryable_groq_errors

    def test_includes_connection_error(self) -> None:
        assert groq.APIConnectionError in LLMGroqClient.retryable_groq_errors

    def test_includes_timeout_error(self) -> None:
        assert groq.APITimeoutError in LLMGroqClient.retryable_groq_errors

    def test_includes_internal_server_error(self) -> None:
        assert groq.InternalServerError in LLMGroqClient.retryable_groq_errors


@pytest.fixture
def groq_client_with_runnable(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Any], tuple[LLMGroqClient, MagicMock]]:
    """Return a factory that wires a custom astream onto a mock LLMGroqClient.

    Usage::

        client, runnable = groq_client_with_runnable(my_astream_fn)
    """

    def factory(
        astream: Any,  # pyright: ignore[reportAny]
    ) -> tuple[LLMGroqClient, MagicMock]:
        mock_runnable = mocker.MagicMock()
        mock_runnable.astream = astream
        client = _mock_groq_client(mocker)
        monkeypatch.setattr(
            type(client),
            "runnable_lcel",
            property(fget=lambda self: mock_runnable),  # pyright: ignore[reportAny]
        )
        return client, mock_runnable

    return factory


class TestLLMGroqClientAstreamResponse:
    async def test_yields_content_from_chunks(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        chunks = [_make_chunk("Hello"), _make_chunk(" world")]
        client, _ = groq_client_with_runnable(lambda _input: _async_gen(*chunks))

        results = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]
        assert results == ["Hello", " world"]

    async def test_skips_empty_content_chunks(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        chunks = [_make_chunk(""), _make_chunk("data"), _make_chunk("")]
        client, _ = groq_client_with_runnable(lambda _input: _async_gen(*chunks))

        results = [
            c
            async for c in client.astream_response([Document(page_content="ctx")], "q")
        ]
        assert results == ["data"]

    async def test_concatenates_document_context(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk("ok")

        client, _ = groq_client_with_runnable(fake_astream)
        docs = [Document(page_content="Part A"), Document(page_content="Part B")]
        _ = [c async for c in client.astream_response(docs, "q")]
        assert captured["context"] == "Part A\n\nPart B"

    async def test_passes_question_in_payload(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk(content="ans")

        client, _ = groq_client_with_runnable(fake_astream)
        _ = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="my question"
            )
        ]
        assert captured["question"] == "my question"

    async def test_defaults_chat_history_to_empty_list(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk(content="ok")

        client, _ = groq_client_with_runnable(fake_astream)
        _ = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]
        assert captured["chat_history"] == []

    async def test_passes_chat_history_when_provided(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk(content="ok")

        client, _ = groq_client_with_runnable(fake_astream)
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

    async def test_handles_empty_document_list(
        self,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_astream(input: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            captured.update(input)
            yield _make_chunk("ok")

        client, _ = groq_client_with_runnable(fake_astream)
        _ = [c async for c in client.astream_response([], "q")]
        assert captured["context"] == ""

    async def test_retryable_error_is_retried(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        call_count = 0

        async def flaky_astream(_: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise groq.RateLimitError(
                    message="rate limit",
                    response=MagicMock(headers={}),
                    body={},
                )
            yield _make_chunk(content="ok")

        client, _ = groq_client_with_runnable(flaky_astream)
        _ = mocker.patch(
            "backend.core.llms.groq.wait_exponential_jitter",
            return_value=lambda _: 0,
        )
        monkeypatch.setattr(settings.llm, "max_retries", 3, raising=False)

        results = [
            c
            async for c in client.astream_response([Document(page_content="ctx")], "q")
        ]

        assert call_count == 2
        assert "ok" in results

    async def test_non_retryable_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        async def bad_astream(_: dict[str, Any]) -> Never:
            raise ValueError("unexpected")
            yield _make_chunk(content="never")  # pyright: ignore[reportUnreachable]

        client, _ = groq_client_with_runnable(bad_astream)
        monkeypatch.setattr(settings.llm, "max_retries", 1, raising=False)

        with pytest.raises(ValueError, match="unexpected"):
            _ = [
                c
                async for c in client.astream_response(
                    documents=[Document(page_content="ctx")], question="q"
                )
            ]

    async def test_mid_stream_retryable_error_yields_interrupted_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        groq_client_with_runnable: Callable[[Any], tuple[LLMGroqClient, MagicMock]],
    ) -> None:
        error = groq.APIConnectionError(request=MagicMock())
        first_call = True

        async def fake_astream(_: dict[str, Any]) -> AsyncIterator[AIMessageChunk]:
            nonlocal first_call
            if first_call:
                first_call = False
                yield _make_chunk(content="partial")
                raise error
            yield _make_chunk(content="ok")

        client, _ = groq_client_with_runnable(fake_astream)
        monkeypatch.setattr(settings.llm, "max_retries", 1, raising=False)

        results = [
            c
            async for c in client.astream_response(
                documents=[Document(page_content="ctx")], question="q"
            )
        ]
        assert any("Stream interrupted" in r for r in results)
