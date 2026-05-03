# =========================
# 📄 AI PDF Assistant (RAG)
# =========================

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# =========================
# 🎨 UI Setup
# =========================
st.set_page_config(page_title="PDF AI Assistant", page_icon="📄", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>📄 AI PDF Assistant</h1>
<p style='text-align:center;'>Upload PDFs and ask questions</p>
""", unsafe_allow_html=True)

# =========================
# 🧠 Session State
# =========================
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# 📂 Process PDFs (RAG Pipeline)
# =========================
def process_pdfs(uploaded_files):
    docs = []

    # Save + Load PDFs
    os.makedirs("docs", exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join("docs", file.name)

        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings (FREE + FAST)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = InMemoryVectorStore.from_documents(
        split_docs,
        embedding=embeddings
    )

    # LLM (Groq)
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    return qa_chain

# =========================
# 📂 Sidebar (Upload)
# =========================
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            st.session_state.qa_chain = process_pdfs(uploaded_files)
        st.success("✅ PDFs processed successfully!")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# 💬 Chat UI
# =========================
if st.session_state.qa_chain:

    st.markdown("### 💬 Chat with your PDF")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask something from your PDF...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.qa_chain.invoke({"query": user_input})
            answer = response["result"]

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload a PDF to start chatting")