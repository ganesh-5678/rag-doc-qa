---
title: RAG Document Q&A
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# RAG Document Q&A — Powered by LLaMA3 & FAISS

Upload any PDF or TXT document and ask questions about its content in natural language. The system retrieves the most relevant sections and generates accurate, context-grounded answers using LLaMA3 via Groq.

## How It Works

1. **Upload** a PDF or TXT document via the sidebar
2. The document is split into overlapping chunks and embedded using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key needed for embeddings)
3. Chunks are indexed into a **FAISS** vector store for fast similarity search
4. When you ask a question, the top-4 most relevant chunks are retrieved
5. Retrieved context + question are passed to **LLaMA3** (via Groq API) to generate a grounded answer
6. Source chunks used to generate the answer are shown for transparency

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | LLaMA3-8B / LLaMA3-70B via Groq API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS (CPU) |
| RAG Framework | LangChain |
| Document Loading | PyPDF, TextLoader |
| UI | Streamlit |
| Deployment | Docker → Hugging Face Spaces |

## Project Structure

```text
.
├── app.py                        # Streamlit UI and app logic
├── src/
│   ├── __init__.py
│   ├── document_processor.py     # PDF/TXT loading and chunking
│   ├── vector_store.py           # FAISS index creation and retrieval
│   └── rag_chain.py              # Groq LLaMA3 prompt and chain
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## Run Locally

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-doc-qa
cd rag-doc-qa
pip install -r requirements.txt
```

### 2. Set up your Groq API key

```bash
cp .env.example .env
# Edit .env and add your key from https://console.groq.com
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Deploy to Hugging Face Spaces

```bash
hf auth login
hf repos create YOUR_USERNAME/rag-doc-qa --type space --space-sdk docker
hf upload YOUR_USERNAME/rag-doc-qa . --type space
```

The app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/rag-doc-qa
```

> **Note:** On Hugging Face Spaces, users enter their own Groq API key in the sidebar. The key is never stored or logged.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 800 | Characters per chunk |
| `chunk_overlap` | 100 | Overlap between chunks |
| `top_k` | 4 | Chunks retrieved per query |
| `temperature` | 0.2 | LLM response temperature |
| `max_tokens` | 1024 | Max tokens in response |

## Resume Line

Built a RAG-based document Q&A system using LangChain and FAISS vector store with LLaMA3 inference via Groq API; supports PDF and TXT uploads with chunk-level source attribution, deployed as a Streamlit app on Hugging Face Spaces via Docker.
