import streamlit as st
from groq import AuthenticationError

from src import (
    ask,
    build_vector_store,
    get_llm,
    load_document,
    retrieve_chunks,
    split_documents,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3147;
    }

    /* Chat messages */
    .user-msg {
        background: #1e3a5f;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .assistant-msg {
        background: #1a2535;
        border-left: 3px solid #10b981;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .user-label   { color: #3b82f6; }
    .assist-label { color: #10b981; }

    /* Source card */
    .source-card {
        background: #111827;
        border: 1px solid #2e3147;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.80rem;
        color: #94a3b8;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1d27;
        border: 1px solid #2e3147;
        border-radius: 8px;
        padding: 12px;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1a1d27 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2e3147 !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #2563eb;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "llm": None,
        "doc_name": None,
        "chunk_count": 0,
        "api_key_valid": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Document Q&A")
    st.markdown("*Powered by LLaMA3 · Groq · FAISS*")
    st.divider()

    # API Key
    st.markdown("### 🔑 Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    # Model selector
    st.markdown("### 🤖 Model")
    model_choice = st.selectbox(
        "Select LLaMA3 model",
        options=  ["llama-3.1-8b-instant","llama-3.3-70b-versatile"],
        index=0,
        help="70b is more accurate but slower on free tier.",
    )

    st.divider()

    # File uploader
    st.markdown("### 📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Supported: PDF, TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file and api_key:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Loading and indexing document..."):
                try:
                    docs   = load_document(uploaded_file)
                    chunks = split_documents(docs)
                    vs     = build_vector_store(chunks)
                    llm    = get_llm(api_key, model=model_choice)

                    st.session_state.vector_store  = vs
                    st.session_state.llm           = llm
                    st.session_state.doc_name      = uploaded_file.name
                    st.session_state.chunk_count   = len(chunks)
                    st.session_state.chat_history  = []
                    st.session_state.api_key_valid = True

                    st.success(f"✅ Indexed {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    elif uploaded_file and not api_key:
        st.warning("Enter your Groq API key first.")
    elif api_key and not uploaded_file:
        st.info("Upload a document to get started.")

    st.divider()

    # Stats
    if st.session_state.vector_store:
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.chunk_count)
        col2.metric("Model", model_choice.split("-")[1])
        st.caption(f"📄 {st.session_state.doc_name}")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<div style='color:#475569;font-size:0.75rem;text-align:center;margin-top:12px'>"
        "Built with LangChain · Groq · FAISS<br>"
        "<a href='https://console.groq.com' style='color:#3b82f6'>Get free Groq API key →</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 📄 RAG Document Q&A")
st.markdown("Upload any PDF or TXT document and ask questions about its content.")
st.divider()

# Empty state
if not st.session_state.vector_store:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#475569;">
        <div style="font-size:3rem;margin-bottom:16px">📂</div>
        <div style="font-size:1.1rem;font-weight:600;color:#94a3b8;margin-bottom:8px">
            No document loaded
        </div>
        <div style="font-size:0.9rem">
            Add your Groq API key and upload a PDF or TXT file in the sidebar to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Chat history display ───────────────────────────────────────────────────────
for turn in st.session_state.chat_history:
    st.markdown(
        f'<div class="user-msg">'
        f'<div class="label user-label">You</div>{turn["question"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="assistant-msg">'
        f'<div class="label assist-label">Assistant</div>{turn["answer"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if turn.get("sources"):
        with st.expander(f"📎 {len(turn['sources'])} source chunks used", expanded=False):
            for i, src in enumerate(turn["sources"], 1):
                page = src.metadata.get("page", "N/A")
                preview = src.page_content[:220].replace("\n", " ") + "..."
                st.markdown(
                    f'<div class="source-card"><b>Chunk {i}</b> · Page {page}<br>{preview}</div>',
                    unsafe_allow_html=True,
                )

# ── Question input ─────────────────────────────────────────────────────────────
st.divider()
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Ask a question about your document",
        placeholder="e.g. What is the main topic of this document?",
        label_visibility="collapsed",
        key="question_input",
    )
with col_btn:
    ask_clicked = st.button("Ask →", use_container_width=True)

# ── Answer generation ──────────────────────────────────────────────────────────
if ask_clicked and question.strip():
    with st.spinner("Searching document and generating answer..."):
        try:
            retrieved = retrieve_chunks(st.session_state.vector_store, question)
            answer    = ask(st.session_state.llm, retrieved, question)

            st.session_state.chat_history.append({
                "question": question,
                "answer":   answer,
                "sources":  retrieved,
            })
            st.rerun()

        except AuthenticationError:
            st.error("❌ Invalid Groq API key. Please check and re-enter.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

elif ask_clicked and not question.strip():
    st.warning("Please enter a question.")
