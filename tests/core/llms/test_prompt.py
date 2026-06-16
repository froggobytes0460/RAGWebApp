from langchain_core.messages import HumanMessage, SystemMessage

from backend.core.llms.prompt import RAG_PROMPT, RAG_SYSTEM_PROMPT


class TestRAGSystemPrompt:
    def test_contains_context_placeholder(self) -> None:
        assert "{context}" in RAG_SYSTEM_PROMPT

    def test_contains_fallback_answer(self) -> None:
        assert "I don't know based on the provided information." in RAG_SYSTEM_PROMPT

    def test_contains_instructions(self) -> None:
        assert "Instructions:" in RAG_SYSTEM_PROMPT

    def test_context_is_wrapped_in_xml_tags(self) -> None:
        assert "<context>" in RAG_SYSTEM_PROMPT
        assert "</context>" in RAG_SYSTEM_PROMPT


class TestRAGPromptTemplate:
    def test_input_variables_are_correct(self) -> None:
        # chat_history uses MessagesPlaceholder(optional=True), so LangChain excludes it
        # from required input_variables but it is still accepted as an optional input
        assert set(RAG_PROMPT.input_variables) == {"context", "question"}

    def test_renders_context_and_question(self) -> None:
        messages = RAG_PROMPT.format_messages(
            context="Some context text.",
            question="What is this?",
            chat_history=[],
        )
        combined = " ".join(str(m.content) for m in messages)
        assert "Some context text." in combined
        assert "What is this?" in combined

    def test_renders_chat_history_messages(self) -> None:
        history = [HumanMessage(content="Hi"), SystemMessage(content="Hello")]
        messages = RAG_PROMPT.format_messages(
            context="ctx",
            question="q",
            chat_history=history,
        )
        contents = [str(m.content) for m in messages]
        assert "Hi" in contents
        assert "Hello" in contents

    def test_renders_with_empty_chat_history(self) -> None:
        messages = RAG_PROMPT.format_messages(
            context="ctx",
            question="q",
            chat_history=[],
        )
        assert len(messages) >= 2

    def test_chat_history_is_optional(self) -> None:
        messages = RAG_PROMPT.format_messages(
            context="ctx",
            question="q",
        )
        assert len(messages) >= 2
