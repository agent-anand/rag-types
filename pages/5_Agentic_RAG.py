# pages/5_Agentic_RAG.py

# Streamlit page for Agentic RAG functionality
# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate # For react agent
from langchain import hub # To pull react prompt
from utils import get_llm, load_vectorstore_from_utils

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("🤖 Agentic RAG")

# Load vectorstore and retriever
vectorstore = load_vectorstore_from_utils()
retriever = vectorstore.as_retriever()

# Define a function for Agentic RAG
# This function uses an agent to retrieve and process documents
def agentic_rag(query):
    _llm = get_llm()
    # Use the LLM to generate a response
    response = _llm.invoke(f"Answer the following question: {query}")
    return response.content

# Input field for user query
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        # Perform Agentic RAG and display the result
        result = agentic_rag(query)
        st.write(result)