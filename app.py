# =========================
# 📄 AI PDF Assistant (STABLE VERSION)
# =========================

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# =========================
# UI
# =========================
st.set_page_config(page_title="PDF AI Assistant", page_icon="📄", layout="wide")

st.title("📄 AI PDF Assistant")

# =========================
# Session State
# =========================
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# PROCESS PDF
# =========================
def process_pdfs(uploaded_files):
    docs = []

    os.makedirs("docs", exist_ok=True)

    for file in uploaded_files:
        path = os.path.join("docs", file.name)

        with open(path, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store (FAISS = better)
    vector_db = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0
    )

    # RAG Chain (OLD but stable)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    return qa_chain

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Upload PDFs")

    files = st.file_uploader(
        "Upload",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:
        with st.spinner("Processing..."):
            st.session_state.qa_chain = process_pdfs(files)
        st.success("Done!")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# CHAT
# =========================
if st.session_state.qa_chain:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask from PDF...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"query": query})

            answer = response["result"]
            sources = response["source_documents"]

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("Sources"):
                for doc in sources:
                    st.write(doc.page_content[:300])

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload PDF to start")