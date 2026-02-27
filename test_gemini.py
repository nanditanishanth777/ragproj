from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

response = llm.invoke("What is a RAG application in simple terms?")
print(response.content)
