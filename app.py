import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📚 Study Assistant")
    st.markdown("---")
    st.markdown("### Upload your notes")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown("---")
    st.markdown("### Sample Questions")
    st.markdown("- What is a microcontroller?")
    st.markdown("- Explain 8051 architecture")
    st.markdown("- What are interrupts?")
    st.markdown("- Explain memory organization")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main area
st.title("📚 Study Assistant")
st.caption("Powered by RAG — answers from your own study notes!")
st.markdown("---")

def load_rag_pipeline(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db3")

    llm = ChatGroq(model="llama-3.1-8b-instant")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful study assistant for students.
    Use the context below to answer the question.
    If the context has relevant information, use it.
    If not, use your general knowledge to help the student.
    Always give a clear, helpful answer.

    Context: {context}
    Question: {question}

    Give a detailed, student-friendly answer:
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain, len(pages)

# Load pipeline
if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            chain, num_pages = load_rag_pipeline(tmp_path)
            st.session_state.chain = chain
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            os.unlink(tmp_path)
            st.success(f"✅ Loaded {uploaded_file.name} — {num_pages} pages ready!")

else:
    # Fall back to default notes.pdf
    if "chain" not in st.session_state:
        with st.spinner("Loading default notes..."):
            chain, num_pages = load_rag_pipeline("notes.pdf")
            st.session_state.chain = chain
            st.session_state.current_file = "notes.pdf"
            st.info(f"📄 Using default notes.pdf — {num_pages} pages")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
            st.write(answer.content)

    st.session_state.messages.append({"role": "assistant", "content": answer.content})