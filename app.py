# =========================
# 📄 AI PDF Assistant (PRO RAG VERSION)
# =========================

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

from langchain_groq import ChatGroq

# NEW RAG CHAINS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# =========================
# 🎨 UI
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
# 📂 PROCESS PDFs
# =========================
def process_pdfs(uploaded_files):
    docs = []

    os.makedirs("docs", exist_ok=True)

    # Load PDFs
    for file in uploaded_files:
        file_path = os.path.join("docs", file.name)

        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings (Best free model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = InMemoryVectorStore.from_documents(
        split_docs,
        embedding=embeddings
    )

    # LLM
    llm = ChatGroq(
        model="llama3-8b-8192",  # fast + stable
        temperature=0
    )

    # Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    # Prompt (STRICT)
    prompt = ChatPromptTemplate.from_template("""
You are a PDF assistant.

Rules:
1. Answer ONLY from the given context
2. If answer is not in context, say: "Not found in document"
3. Keep answer clear and structured

Context:
{context}

Question:
{input}
""")

    # Chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain

# =========================
# 📂 SIDEBAR
# =========================
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            st.session_state.qa_chain = process_pdfs(uploaded_files)
        st.success("✅ PDFs processed!")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# 💬 CHAT UI
# =========================
if st.session_state.qa_chain:

    st.markdown("### 💬 Chat with your PDF")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask something from your PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤖 Thinking..."):
            response = st.session_state.qa_chain.invoke({"input": user_input})

            answer = response["answer"]
            sources = response.get("context", [])

        with st.chat_message("assistant"):
            st.markdown(answer)

            # 🔍 Show sources
            with st.expander("📄 Source Context"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500])

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload PDF to start chatting")