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


QUERY_GEN_SYSTEM_PROMPT = """You are a search query optimiser for a document retrieval system.

Given a user question and optional chat history, produce a JSON object with exactly two keys:
- "query": a concise, keyword-rich search string optimised for vector similarity retrieval (no question marks, no filler words).
- "filters": an object with optional keys "filename" (string), "uploaded_after" (ISO-8601 UTC datetime string), "uploaded_before" (ISO-8601 UTC datetime string). Set each to null if not applicable.

Return ONLY the raw JSON object. Do not include markdown, explanation, or any other text."""

QUERY_GEN_PROMPT = ChatPromptTemplate(
    messages=[
        ("system", QUERY_GEN_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ],
    input_types={"chat_history": list[BaseMessage], "question": str},
)

RAG_PROMPT = ChatPromptTemplate(
    messages=[
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ],
    input_types={"context": str, "chat_history": list[BaseMessage], "question": str},
)
