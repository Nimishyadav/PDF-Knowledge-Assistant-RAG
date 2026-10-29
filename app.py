"""Simple document RAG: upload, index once, and chat with sources."""
import hashlib
import importlib
import io
import json

import streamlit as st
from pypdf import PdfReader

from src.chunker import chunk_text
from src import llm
from src.retriever import retrieve
from src.vector_store import VectorStore

# Streamlit can retain imported modules across script reruns after a code update.
# Refresh this lightweight wrapper so the UI and backend use the same API.
llm = importlib.reload(llm)

st.set_page_config(page_title="Local PDF Knowledge Assistant (RAG)", page_icon="📚", layout="wide")

@st.cache_resource
def get_embedder():
    from src.embedder import Embedder
    return Embedder()


def build_documents(files, size, overlap):
    documents = []
    for file in files:
        if file.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file.getvalue()))
            pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        else:
            pages = [(1, file.getvalue().decode("utf-8-sig"))]
        for page, text in pages:
            for chunk in chunk_text(text.strip(), size, overlap):
                if chunk.strip():
                    documents.append({"text": chunk, "source": file.name, "page": page})
    return documents


st.title("📚 Local PDF Knowledge Assistant (RAG)")
st.caption("Upload text-based PDFs or UTF-8 TXT files. Answers use retrieved document passages through local Ollama.")
for key, default in {"messages": [], "documents": [], "index": None, "signature": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF / TXT", type=["pdf", "txt"], accept_multiple_files=True)
    size = st.slider("Chunk size (characters)", 200, 2000, 800, 100)
    overlap = st.slider("Overlap (characters)", 0, 400, 100, 50)
    top_k = st.slider("Retrieved passages", 1, 10, 4)
    model = st.text_input("Ollama model", "llama3").strip() or "llama3"
    cpu_only = st.checkbox("Use CPU (avoids CUDA errors)", value=True)
    st.caption("CPU mode can be slower. Disable it to try GPU; CUDA failures retry on CPU.")
    signature = hashlib.sha256(repr([(f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files] + [size, overlap]).encode()).hexdigest()
    if st.button("Process documents", type="primary", disabled=not files):
        if overlap >= size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            try:
                with st.spinner("Extracting text and creating embeddings…"):
                    documents = build_documents(files, size, overlap)
                    if not documents:
                        raise ValueError("No text found. Scanned PDFs need OCR; use a text-based PDF or TXT file.")
                    embeddings = get_embedder().encode([d["text"] for d in documents])
                    store = VectorStore(embeddings.shape[1])
                    store.add(embeddings)
                    st.session_state.update(documents=documents, index=store.index, signature=signature, messages=[])
                st.success("Documents indexed. Ready to chat!")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
    if st.session_state.index is not None:
        st.metric("Indexed passages", len(st.session_state.documents))
        st.write("Indexed files:", ", ".join(dict.fromkeys(d["source"] for d in st.session_state.documents)))
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Reset documents and chat"):
        st.session_state.update(documents=[], index=None, signature=None, messages=[])
        st.rerun()
    if st.session_state.messages:
        st.download_button("Download chat", json.dumps(st.session_state.messages, ensure_ascii=False, indent=2), "rag-chat.json", "application/json")

ready = st.session_state.index is not None and signature == st.session_state.signature
if st.session_state.index is not None and not ready:
    st.warning("Uploads or chunk settings changed. Click Process documents before chatting.")
elif not ready:
    st.info("Upload documents in the sidebar, then click Process documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources / retrieved context"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**[{i}] {doc['source']} — page {doc['page']}**")
                    st.text(doc["text"])

question = st.chat_input("Ask about your documents", disabled=not ready)
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Finding passages and generating answer…"):
                # Include recent turns in retrieval to help resolve follow-up questions.
                previous = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-2:]
                query = "\n".join(previous + [question])
                vector = get_embedder().encode([query])[0]
                sources = retrieve(vector, st.session_state.index, st.session_state.documents, top_k)
                context = "\n\n".join(f"[{i}] {d['source']} (page {d['page']})\n{d['text']}" for i, d in enumerate(sources, 1))
                answer = llm.ask_llm(question, context, model=model, history=st.session_state.messages[-6:], cpu_only=cpu_only)
            st.markdown(answer)
            with st.expander("Sources / retrieved context"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {doc['source']} — page {doc['page']}**")
                    st.text(doc["text"])
        st.session_state.messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer, "sources": sources}])
    except Exception as exc:
        st.error(f"Could not answer: {exc}. Check that Ollama is running and the selected model is installed, then retry.")


