from langchain_core.messages import HumanMessage, SystemMessage

from backend.core.llms.prompt import (
    HYPE_PROMPT,
    HYPE_SYSTEM_PROMPT,
    RAG_PROMPT,
    RAG_SYSTEM_PROMPT,
)


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


class TestHyPESystemPrompt:
    def test_contains_n_placeholder(self) -> None:
        assert "{n}" in HYPE_SYSTEM_PROMPT

    def test_contains_json_array_instruction(self) -> None:
        assert "JSON array" in HYPE_SYSTEM_PROMPT

    def test_contains_diversity_rule(self) -> None:
        assert "diverse" in HYPE_SYSTEM_PROMPT


class TestHyPEPromptTemplate:
    def test_input_variables_are_correct(self) -> None:
        assert "chunk" in HYPE_PROMPT.input_variables

    def test_renders_chunk_in_human_message(self) -> None:
        messages = HYPE_PROMPT.format_messages(chunk="Some passage text.", n=3)
        human_contents = [str(m.content) for m in messages if m.type == "human"]
        assert any("Some passage text." in c for c in human_contents)

    def test_renders_n_in_system_message(self) -> None:
        messages = HYPE_PROMPT.format_messages(chunk="passage", n=5)
        system_contents = [str(m.content) for m in messages if m.type == "system"]
        assert any("5" in c for c in system_contents)
