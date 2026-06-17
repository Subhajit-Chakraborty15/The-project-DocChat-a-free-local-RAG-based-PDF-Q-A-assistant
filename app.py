"""
DocChat - A RAG-based PDF Question Answering Assistant
--------------------------------------------------------
Upload a PDF, ask questions about it, get answers grounded in the document.

Pipeline (this is the "RAG" - Retrieval-Augmented Generation - pattern):
1. Load PDF and split into small text chunks
2. Convert each chunk into a vector embedding (numeric representation of meaning)
3. Store embeddings in a FAISS vector index (local, free, fast similarity search)
4. When user asks a question -> embed the question -> find most similar chunks
5. Send those chunks + the question to an LLM (via Groq's free API) -> get an answer

Everything here is free:
- sentence-transformers: runs locally, no API key, no cost
- FAISS: local vector store, no API key, no cost
- Groq: free-tier LLM API, no credit card required to sign up
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Get a free Groq API key at https://console.groq.com (no card required)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free, local
LLM_MODEL = "llama-3.1-8b-instant"  # free tier on Groq

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_CHUNKS = 4  # how many chunks to retrieve per question

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a specific document.
Use ONLY the context below to answer. If the answer isn't in the context, say you don't know -
do not make things up.

Context:
{context}

Question: {question}

Answer (be concise and specific):"""


# ---------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the local embedding model once and cache it across reruns."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_split_pdf(file_path: str):
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_vector_store(chunks):
    """Embed all chunks and build a FAISS index for similarity search."""
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_qa_chain(vector_store):
    """Wire up the retriever + LLM into a question-answering chain."""
    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY found. Set it as an environment variable before running.")
        st.stop()

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0.1,  # low temperature = more factual, less creative
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DocChat - RAG PDF Q&A", page_icon="📄")
st.title("📄 DocChat")
st.caption("Ask questions about any PDF. Answers are grounded in the document using RAG.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading and indexing document..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                chunks = load_and_split_pdf(temp_path)
                vector_store = build_vector_store(chunks)
                st.session_state.qa_chain = build_qa_chain(vector_store)
                st.session_state.chat_history = []

                os.remove(temp_path)

            st.success(f"Indexed {len(chunks)} chunks. Ready to chat!")

    st.divider()
    st.caption(
        "Stack: LangChain · sentence-transformers (local embeddings) · "
        "FAISS (local vector store) · Groq (free Llama 3 inference)"
    )

st.header("2. Ask questions")

if st.session_state.qa_chain is None:
    st.info("Upload and process a PDF in the sidebar to get started.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    question = st.chat_input("Ask something about the document...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])

                st.write(answer)

                if sources:
                    with st.expander("Source chunks used"):
                        for i, doc in enumerate(sources, start=1):
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**Chunk {i} (page {page}):**")
                            st.text(doc.page_content[:300] + "...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
