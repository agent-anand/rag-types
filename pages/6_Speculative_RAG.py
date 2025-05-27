# Streamlit page for Speculative RAG functionality

# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from utils import get_llm, load_vectorstore_from_utils

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("🔮 Speculative RAG")

# Load vectorstore and retriever
vectorstore = load_vectorstore_from_utils()
retriever = vectorstore.as_retriever()

# Define a function for Speculative RAG
# This function generates multiple candidate answers and scores them based on relevance
def speculative_rag(query):
    _llm = get_llm()
    # Generate candidate answers
    candidates = [
        _llm.invoke(f"Provide a short answer to: {query}").content,
        _llm.invoke(f"Another possible answer to: {query}").content,
        _llm.invoke(f"Based on general knowledge, a concise answer to '{query}' could be:").content
    ]

    scored_candidates = []
    for candidate in candidates:
        # Retrieve documents relevant to the candidate answer
        docs = retriever.invoke(candidate)
        # Score the candidate based on the number of relevant documents
        score = sum(1 for doc in docs if query.lower() in doc.page_content.lower())
        scored_candidates.append((candidate, score))

    # Select the best candidate based on the highest score
    best_answer = max(scored_candidates, key=lambda x: x[1])[0]
    return best_answer

# Input field for user query
user_query = st.text_input("Ask a question:")
if user_query:
    with st.spinner("Speculating and thinking..."):
        # Perform Speculative RAG and display the result
        st.write(speculative_rag(user_query))