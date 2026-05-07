from typing import List

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question using ONLY the context provided below.

Rules:
- If the answer is clearly present in the context, give a direct and complete answer.
- If the context partially answers the question, answer what you can and say what is missing.
- If the answer is not in the context at all, say: "I could not find this in the uploaded document."
- Never make up information not present in the context.
- Keep answers concise and well-structured.

Context:
{context}
"""

HUMAN_PROMPT = "Question: {question}"


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


def get_llm(api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> ChatGroq:
    """
    Initialise the Groq LLaMA3 chat model.
    """
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=1024,
    )


def format_context(docs: List[Document]) -> str:
    """
    Concatenate retrieved chunks into a single context string.
    """
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'document')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )


def ask(llm: ChatGroq, docs: List[Document], question: str) -> str:
    """
    Run the RAG chain: format context → build prompt → call LLM → return answer.
    """
    context = format_context(docs)
    prompt = build_prompt()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
