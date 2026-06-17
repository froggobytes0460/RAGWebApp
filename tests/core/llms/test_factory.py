# pyright: reportPrivateUsage=none
# pyright: reportFunctionMemberAccess=none

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.core.config import settings
from backend.core.llms import LLMClientFactory, LLMClientProto
from backend.core.llms.groq import LLMGroqClient
from backend.core.llms.openrouter import LLMOpenRouterClient


class TestLLMClientFactoryRegistry:
    def test_registry_contains_groq(self) -> None:
        assert "groq" in LLMClientFactory._registry

    def test_registry_contains_openrouter(self) -> None:
        assert "openrouter" in LLMClientFactory._registry

    def test_groq_entry_is_callable(self) -> None:
        assert callable(LLMClientFactory._registry["groq"])

    def test_openrouter_entry_is_callable(self) -> None:
        assert callable(LLMClientFactory._registry["openrouter"])

    def test_groq_entry_name(self) -> None:
        assert (
            LLMClientFactory._registry["groq"].__func__  # pyright: ignore[reportAny]
            is LLMGroqClient.from_settings.__func__
        )

    def test_openrouter_entry_name(self) -> None:
        assert (
            LLMClientFactory._registry[
                "openrouter"
            ].__func__  # pyright: ignore[reportAny]
            is LLMOpenRouterClient.from_settings.__func__
        )


class TestLLMClientFactoryFromSettings:
    def test_returns_groq_client(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings.llm, "provider", "groq", raising=False)
        mock_instance = mocker.MagicMock(spec=LLMGroqClient)
        original = LLMClientFactory._registry.copy()
        LLMClientFactory._registry["groq"] = lambda: mock_instance
        try:
            result = LLMClientFactory.from_settings()
            assert result is mock_instance
        finally:
            LLMClientFactory._registry.clear()
            LLMClientFactory._registry.update(original)

    def test_returns_openrouter_client(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings.llm, "provider", "openrouter", raising=False)
        mock_instance = mocker.MagicMock(spec=LLMOpenRouterClient)
        original = LLMClientFactory._registry.copy()
        LLMClientFactory._registry["openrouter"] = lambda: mock_instance
        try:
            result = LLMClientFactory.from_settings()
            assert result is mock_instance
        finally:
            LLMClientFactory._registry.clear()
            LLMClientFactory._registry.update(original)

    def test_raises_value_error_for_unknown_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.llm, "provider", "unknown_llm", raising=False)

        with pytest.raises(ValueError, match="Unsupported LLM provider: unknown_llm"):
            _ = LLMClientFactory.from_settings()

    def test_raises_type_error_when_client_does_not_satisfy_protocol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.llm, "provider", "groq", raising=False)

        class _Incomplete:
            pass

        original = LLMClientFactory._registry.copy()
        LLMClientFactory._registry["groq"] = (  # pyright: ignore[reportArgumentType]
            _Incomplete
        )

        try:
            with pytest.raises(TypeError, match="does not fully implement"):
                _ = LLMClientFactory.from_settings()
        finally:
            LLMClientFactory._registry.clear()
            LLMClientFactory._registry.update(original)


class TestLLMClientProto:
    def test_groq_client_satisfies_protocol(self, mocker: MockerFixture) -> None:
        mock = mocker.MagicMock(spec=LLMGroqClient)
        assert isinstance(mock, LLMClientProto)

    def test_openrouter_client_satisfies_protocol(self, mocker: MockerFixture) -> None:
        mock = mocker.MagicMock(spec=LLMOpenRouterClient)
        assert isinstance(mock, LLMClientProto)

    def test_arbitrary_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(MagicMock(), LLMClientProto)

    def test_object_without_astream_response_does_not_satisfy_protocol(self) -> None:
        class _NoStream:
            @classmethod
            def from_settings(cls) -> "_NoStream":
                return cls()

        assert not isinstance(_NoStream(), LLMClientProto)

    def test_object_without_from_settings_does_not_satisfy_protocol(self) -> None:
        class _NoFactory:
            async def astream_response(self) -> None:
                pass

        assert not isinstance(_NoFactory(), LLMClientProto)
