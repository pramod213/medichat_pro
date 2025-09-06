from langchain_community.vectorstores import FAISS 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv 
from typing import List
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def create_faiss_index(texts: List[str]):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(texts, embeddings)

def retrieve_relevant_docs(vectorstore: FAISS, query: str, k: int = 4):
    return vectorstore.similarity_search(query, k=k)
