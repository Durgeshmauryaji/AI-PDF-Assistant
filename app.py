from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

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




# Data in st session state
# While file not uploaded, document_uploaded is False and agent is None. Once file is uploaded, document_uploaded becomes True and agent is created.
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False

if "agent" not in st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "messages" not in st.session_state:
    st.session_state.messages=[]

# Document processing 
def process_document(file):
    """Processes the uploaded PDF document and creates the agent."""
    # Load the PDF document
    loader=PyPDFLoader(file)
    docs=loader.load()

    # Split the document into smaller chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)

    # Embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs, 
        embedding=embeddings
    )

    # Create a Agent--LLM, Tool and prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query:str):
        """Retrieves relevant context from the vector store based on the query."""
        context=""
        docs=vector_db.similarity_search(query=query,k=4)
        for doc in docs:
            context+=doc.page_content
        return context    

    system_prompt = """You are a helpful assistant.
    ALWAYS use the retrieve_context tool to get relevant information from the document before answering.
    Do NOT answer from your own knowledge.
    After retrieving context, answer strictly based on it.
    """

    memory=InMemorySaver()

    agent=create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent=agent
    st.session_state.document_uploaded=True


# Upload file and ask questions in Streamlit
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded and not st.session_state.document_uploaded:
        with st.spinner("Processing..."):
            path = "./docs_files"

            import os
            os.makedirs(path, exist_ok=True)

            for file in uploaded:
                with open(f"{path}/{file.name}", "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.success("✅ Documents processed!")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()




# Chat interface
if st.session_state.document_uploaded and st.session_state.agent:

    st.markdown("### 💬 Chat with your document")

    # Chat container
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

    # Input box
    user_input = st.chat_input("Ask something from your PDF...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.chat_message("user").markdown(user_input)

        with st.spinner("🤖 Thinking..."):

            response = st.session_state.agent.invoke(
                {"messages":[{"role":"user","content":user_input}]},
                {"configurable":{"thread_id":1}}
            )

            answer = response["messages"][-1].content

            st.chat_message("assistant").markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
    