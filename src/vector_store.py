from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer embedding model.
    Runs locally — no API key required.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector store from document chunks.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def retrieve_chunks(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
