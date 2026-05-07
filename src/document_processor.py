import os
import tempfile
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(uploaded_file) -> List[Document]:
    """
    Load a PDF or TXT file uploaded via Streamlit and return a list of Documents.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Upload a PDF or TXT file.")

        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    return documents


def split_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """
    Split documents into overlapping chunks for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks
