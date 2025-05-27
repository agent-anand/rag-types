# Streamlit page for Simple RAG functionality

# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from utils import get_llm, load_vectorstore_from_utils

# Load environment variables from .env file
load_dotenv()

# Initialize vectorstore and retriever if not already in session state
if "qa_chain" not in st.session_state:
    # Load vectorstore using utility function
    vectorstore = load_vectorstore_from_utils()
    retriever = vectorstore.as_retriever()
    # Initialize QA chain and store in session state
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever
    )

# Streamlit app title
st.title("🔹 Simple RAG")

# Input field for user query
query = st.text_input("Ask a question based on your document:")
if query:
    with st.spinner("Thinking..."):
        # Retrieve and display the result from the QA chain
        result = st.session_state.qa_chain.invoke({"query": query})
        st.write(result["result"])