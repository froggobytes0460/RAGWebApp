from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = """You are an expert assistant tasked with answering questions using only the provided context.

Instructions:
1. Base your answer strictly on the text inside the <context> tags.
2. Do not assume, extrapolate, or bring in outside knowledge.
3. Whenever possible, reference or quote the specific part of the context you used to form your answer.
4. If the context does not explicitly contain the answer, respond exactly with: "I don't know based on the provided information."
5. Keep your answer factual, direct, and concise.

<context>
{context}
</context>"""

RAG_PROMPT = ChatPromptTemplate(
    messages=[
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ],
    input_types={"context": str, "chat_history": list[BaseMessage], "question": str},
)

HYPE_SYSTEM_PROMPT = """You are a question generation assistant for a document retrieval system.

Given a text passage, generate exactly {n} distinct, specific questions that this passage directly answers.

Rules:
- Each question must be answerable solely from the passage.
- Questions must be diverse — cover different aspects of the passage.
- Return ONLY a JSON array of strings. No markdown, no explanation.

Example output: ["What is X?", "How does Y work?", "When did Z occur?"]"""

HYPE_PROMPT = ChatPromptTemplate(
    messages=[
        ("system", HYPE_SYSTEM_PROMPT),
        ("human", "{chunk}"),
    ],
    input_types={"chunk": str, "n": int},
)
