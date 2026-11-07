import os
import html
import hashlib
import shutil

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = "uploads"
DB_DIR = "vector_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Custom CSS (dark, modern)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b0f1a;
    --panel: #111827;
    --panel-2: #161e31;
    --border: rgba(148, 163, 184, 0.16);
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #6366f1;
    --accent-2: #22d3ee;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(99,102,241,0.18), transparent 60%),
        radial-gradient(700px 400px at -10% 10%, rgba(34,211,238,0.10), transparent 60%),
        var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container { padding-top: 1.6rem; padding-bottom: 6rem; max-width: 980px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1424 0%, #0b0f1a 100%);
    border-right: 1px solid var(--border);
}
.side-brand { display:flex; align-items:center; gap:.6rem; margin-bottom:1rem; }
.side-logo {
    width:38px; height:38px; border-radius:11px; display:flex;
    align-items:center; justify-content:center; font-size:1.2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
}
.side-name { font-weight:700; font-size:1.05rem; line-height:1.1; }
.side-tag { color: var(--muted); font-size:.75rem; }
.side-label {
    text-transform: uppercase; letter-spacing:.08em; font-size:.68rem;
    color: var(--muted); margin: 1.1rem 0 .4rem 0; font-weight:600;
}

[data-testid="stFileUploader"] section {
    background: var(--panel); border: 1.5px dashed rgba(99,102,241,.5);
    border-radius: 14px; padding: 1rem;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent-2); }

.pdf-card {
    padding: 1rem; border-radius: 14px; margin-top: .6rem;
    background: linear-gradient(145deg, rgba(99,102,241,.16), rgba(34,211,238,.07));
    border: 1px solid rgba(99,102,241,.35);
}
.pdf-name { font-weight:600; font-size:.92rem; word-break: break-word; margin-bottom:.6rem; }
.chips { display:flex; gap:.4rem; flex-wrap:wrap; }
.chip {
    background: rgba(15,23,42,.7); border: 1px solid var(--border);
    border-radius: 999px; padding:.18rem .65rem; font-size:.75rem; color:#cbd5e1;
}

.stButton > button {
    border-radius: 11px; border: 1px solid var(--border);
    background: var(--panel-2); color: var(--text); font-weight: 500;
    transition: all .15s ease;
}
.stButton > button:hover {
    border-color: var(--accent); color: #fff;
    transform: translateY(-1px); box-shadow: 0 6px 18px rgba(99,102,241,.25);
}

/* ---------- Hero ---------- */
.hero { margin-bottom: 1.4rem; }
.hero-title {
    font-size: 2.5rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1;
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); margin-top:.4rem; font-size:1rem; }
.status {
    display:inline-flex; align-items:center; gap:.45rem; margin-top:.9rem;
    padding:.28rem .8rem; border-radius:999px; font-size:.8rem;
    background: var(--panel); border:1px solid var(--border); color:#cbd5e1;
}
.dot { width:8px; height:8px; border-radius:50%; background:#64748b; }
.dot.on { background:#22c55e; box-shadow:0 0 10px #22c55e; }

/* ---------- Empty state ---------- */
.feature {
    background: var(--panel); border:1px solid var(--border); border-radius:16px;
    padding:1.2rem; height:100%; transition: all .2s ease;
}
.feature:hover { border-color: rgba(99,102,241,.55); transform: translateY(-3px); }
.feature .ico { font-size:1.6rem; margin-bottom:.5rem; }
.feature .ttl { font-weight:600; margin-bottom:.25rem; }
.feature .dsc { color: var(--muted); font-size:.85rem; line-height:1.45; }

.steps {
    margin-top:1.4rem; padding:1rem 1.2rem; border-radius:14px;
    background: rgba(99,102,241,.08); border:1px dashed rgba(99,102,241,.4);
    color:#cbd5e1; font-size:.92rem;
}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--panel); border:1px solid var(--border);
    border-radius: 16px; padding: 1rem 1.1rem; margin-bottom:.8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(145deg, rgba(99,102,241,.18), rgba(99,102,241,.08));
    border-color: rgba(99,102,241,.35);
}

[data-testid="stChatInput"] {
    border-radius: 16px; border: 1px solid rgba(99,102,241,.45);
    background: var(--panel);
}
[data-testid="stBottom"] > div { background: transparent; }

/* ---------- Sources ---------- */
.src {
    background: var(--panel-2); border:1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 12px;
    padding:.8rem 1rem; margin-bottom:.6rem;
}
.src-head { display:flex; gap:.45rem; align-items:center; margin-bottom:.45rem; flex-wrap:wrap; }
.badge {
    font-size:.7rem; font-weight:600; padding:.12rem .55rem; border-radius:999px;
    background: rgba(99,102,241,.22); color:#c7d2fe;
}
.badge.page { background: rgba(34,211,238,.18); color:#a5f3fc; }
.badge.dist { background: rgba(148,163,184,.15); color:#cbd5e1; }
.src-body { color:#cbd5e1; font-size:.86rem; line-height:1.55; }

[data-testid="stExpander"] {
    border:1px solid var(--border); border-radius:12px; background: transparent;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background:#243049; border-radius: 8px; }

/* ---------- Force readable colors (works in light or dark base theme) ---------- */
.side-name, .pdf-name { color: #f1f5f9 !important; -webkit-text-fill-color: #f1f5f9 !important; }
.side-tag { color: #94a3b8 !important; }
.chip { color: #e2e8f0 !important; }
.hero-sub { color: #94a3b8 !important; }

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSlider"] p,
[data-testid="stSlider"] div[data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #e5e7eb !important; }

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] {
    background: #111827 !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    caret-color: #ffffff !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

[data-testid="stChatMessage"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { color: #e5e7eb !important; }
.src-body { color: #cbd5e1 !important; }

[data-testid="stFileUploaderFile"] { background: #1e293b !important; border-radius: 10px; }
[data-testid="stFileUploaderFile"] * { color: #e5e7eb !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #e5e7eb !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Cached resources (loaded once, survive Streamlit reruns)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say exactly: "
            "'I could not find the answer in the uploaded document.'\n"
            "Use the conversation history only to understand follow-up questions.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)

# ------------------------------------------------------------------
# Helpers (logic)
# ------------------------------------------------------------------
def to_text(content):
    """Gemini chunks can be a string or a list of parts; always return text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else p.get("text", "") for p in content
        )
    return str(content)


def stream_answer(prompt_value):
    for chunk in get_llm().stream(prompt_value):
        yield to_text(chunk.content)


def build_or_load_db(pdf_path, db_path, embeddings, progress_bar):
    """Load the vector DB for this PDF if it was fully built, else build it.

    Each PDF gets its own folder (named by file hash), so we never need to
    delete a folder that Chroma may still have open -> no WinError 32.
    """
    marker = os.path.join(db_path, ".complete")

    if os.path.exists(marker):
        return Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Leftover from a crashed/interrupted build: nothing has it open now.
    if os.path.isdir(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

    if not chunks:
        raise ValueError(
            "No text could be extracted. This PDF may be scanned images (needs OCR)."
        )

    vs = Chroma(persist_directory=db_path, embedding_function=embeddings)
    batch = 64
    total = len(chunks)
    for i in range(0, total, batch):
        vs.add_documents(chunks[i : i + batch])
        done = min(i + batch, total)
        progress_bar.progress(done / total, text=f"Indexing chunk {done}/{total}")

    with open(marker, "w") as f:
        f.write("ok")
    return vs


def reset_book():
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("pdf_info", None)
    st.session_state.pop("pdf_hash", None)
    st.session_state.messages = []
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


# ------------------------------------------------------------------
# Helpers (UI)
# ------------------------------------------------------------------
def render_sources(sources):
    for i, s in enumerate(sources, start=1):
        body = html.escape(s["text"]).replace("\n", "<br>")
        st.markdown(
            f"""
<div class="src">
  <div class="src-head">
    <span class="badge">Chunk {i}</span>
    <span class="badge page">Page {s['page']}</span>
    <span class="badge dist">distance {s['score']:.3f}</span>
  </div>
  <div class="src-body">{body}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_hero(info):
    if info:
        status = (
            f'<span class="status"><span class="dot on"></span>'
            f'Ready · {html.escape(info["name"])}</span>'
        )
    else:
        status = '<span class="status"><span class="dot"></span>No book loaded</span>'
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-title">📚 RAG Book Assistant</div>
  <div class="hero-sub">Chat with any PDF. Answers come only from your document, with page citations.</div>
  {status}
</div>
""",
        unsafe_allow_html=True,
    )


def render_empty_state():
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📄", "Upload any PDF", "Drag a book, paper or notes into the sidebar. It is indexed once and cached."),
        ("🔎", "Grounded answers", "Each reply is built from the most relevant passages of your document."),
        ("📖", "Cited sources", "Every answer shows the page numbers and exact chunks it used."),
    ]
    for col, (ico, ttl, dsc) in zip((c1, c2, c3), cards):
        col.markdown(
            f"""<div class="feature"><div class="ico">{ico}</div>
            <div class="ttl">{ttl}</div><div class="dsc">{dsc}</div></div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="steps">👈 <b>Get started:</b> upload a PDF from the sidebar, '
        "wait for indexing to finish, then ask your first question.</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

embeddings = get_embeddings()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """<div class="side-brand"><div class="side-logo">📚</div>
        <div><div class="side-name">Book Assistant</div>
        <div class="side-tag">RAG · Gemini · Chroma</div></div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop a PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if st.session_state.get("pdf_info"):
        info = st.session_state.pdf_info
        st.markdown(
            f"""<div class="pdf-card">
            <div class="pdf-name">📘 {html.escape(info['name'])}</div>
            <div class="chips">
              <span class="chip">📑 {info['pages']} pages</span>
              <span class="chip">💾 {info['size_mb']:.2f} MB</span>
            </div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", 2, 8, 4)

    st.markdown('<div class="side-label">Actions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🗑️ New book", use_container_width=True):
        reset_book()
        st.rerun()
    if c2.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Process a newly uploaded PDF (only when it actually changes)
# ------------------------------------------------------------------
if uploaded_file is not None:
    data = uploaded_file.getvalue()
    file_hash = hashlib.md5(data).hexdigest()

    if st.session_state.get("pdf_hash") != file_hash:
        pdf_path = os.path.join(UPLOAD_DIR, f"{file_hash}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(data)

        db_path = os.path.join(DB_DIR, file_hash)
        bar = st.sidebar.progress(0, text="Preparing...")
        try:
            vs = build_or_load_db(pdf_path, db_path, embeddings, bar)
        except Exception as e:
            bar.empty()
            st.sidebar.error(f"Could not process PDF: {e}")
            st.stop()
        bar.empty()

        st.session_state.vectorstore = vs
        st.session_state.pdf_hash = file_hash
        st.session_state.pdf_info = {
            "name": uploaded_file.name,
            "pages": len(PdfReader(pdf_path).pages),
            "size_mb": len(data) / (1024 * 1024),
        }
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
vectorstore = st.session_state.get("vectorstore")
render_hero(st.session_state.get("pdf_info"))

if vectorstore is None:
    render_empty_state()
    st.stop()

# Chat history
for msg in st.session_state.messages:
    avatar = "🙋" if msg["role"] == "user" else "📚"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📖 Sources ({len(msg['sources'])})"):
                render_sources(msg["sources"])

question = st.chat_input("Ask anything about your PDF...")

if question:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)

    results = vectorstore.similarity_search_with_score(question, k=top_k)
    context = "\n\n".join(doc.page_content for doc, _ in results)
    sources = [
        {
            "page": int(doc.metadata.get("page", 0)) + 1,
            "score": float(score),
            "text": doc.page_content,
        }
        for doc, score in results
    ]

    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]
    ) or "(none)"

    prompt_value = PROMPT.invoke(
        {"history": history, "context": context, "question": question}
    )

    with st.chat_message("assistant", avatar="📚"):
        try:
            answer = st.write_stream(stream_answer(prompt_value))
        except Exception as e:
            answer = f"⚠️ The model call failed: {e}"
            st.error(answer)

        with st.expander(f"📖 Sources ({len(sources)})"):
            render_sources(sources)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    