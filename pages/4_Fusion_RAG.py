# Streamlit page for Fusion RAG functionality

# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from utils import get_llm, load_vectorstore_from_utils

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("🔀 Fusion RAG")

# Load vectorstore and retriever
vectorstore = load_vectorstore_from_utils()
retriever = vectorstore.as_retriever()

# Define a function for Fusion RAG
# This function generates sub-queries and retrieves relevant documents for each sub-query
def fusion_rag(query):
    _llm = get_llm()
    # Generate sub-queries using the LLM
    sub_queries_response = _llm.invoke(f"Generate 3 related search queries for: {query}")
    sub_queries = [q.strip() for q in sub_queries_response.content.split("\n") if q.strip()]

    all_docs = []
    for sub_query in sub_queries:
        # Retrieve documents for each sub-query
        docs = retriever.invoke(sub_query)
        all_docs.extend(docs)

    return all_docs

# Input field for user query
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        # Perform Fusion RAG and display results
        documents = fusion_rag(query)
        for doc in documents:
            st.write(doc.page_content)