from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

# ---------------- UI ----------------
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
📄 AI PDF Assistant
</h1>
<p style='text-align: center;'>
Upload your documents and ask intelligent questions
</p>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Process Document ----------------
def process_document(path):

    # Load PDFs
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    # Embeddings (Google)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    # Vector Store
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_db

    # LLM
    llm = ChatGroq(model="openai/gpt-oss-20b")

    # Tool
    @tool
    def retrieve_context(query: str):
        docs = st.session_state.vector_store.similarity_search(query, k=4)
        return "\n".join([doc.page_content for doc in docs])

    # Prompt
    system_prompt = """You are a helpful assistant.
ALWAYS use the retrieve_context tool.
Answer ONLY from document.
If answer not found, say "Not found in document".
"""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        with st.spinner("Processing..."):

            path = "./docs_files"

            # 🧹 IMPORTANT: Purana data delete karo
            if os.path.exists(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))

            os.makedirs(path, exist_ok=True)

            # Save new files
            for file in uploaded:
                with open(f"{path}/{file.name}", "wb") as f:
                    f.write(file.getvalue())

            # Reset old session
            st.session_state.agent = None
            st.session_state.vector_store = None
            st.session_state.messages = []

            # Process new docs
            process_document(path)

            st.success("✅ Documents processed!")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat UI ----------------
if st.session_state.agent:

    st.markdown("### 💬 Chat with your document")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask something from your PDF...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.chat_message("user").markdown(user_input)

        with st.spinner("🤖 Thinking..."):

            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": 1}}
            )

            answer = response["messages"][-1].content

            st.chat_message("assistant").markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })