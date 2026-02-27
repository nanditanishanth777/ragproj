from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Step 1: Loading PDF...")
loader = PyPDFLoader("notes.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

print("Step 3: Creating embeddings and storing in database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Vector database created!")

print("Step 4: Setting up the LLM...")
llm = ChatGroq(model="llama-3.1-8b-instant")
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know, say "I don't have enough information."

Context: {context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("\n✅ RAG Pipeline ready! Ask your questions below.")
print("Type 'exit' to quit\n")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer.content}\n")