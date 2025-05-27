# pages/2_Corrective_RAG.py

# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from utils import get_llm, load_vectorstore_from_utils

# Load environment variables from .env file
load_dotenv()

# Initialize vectorstore and retriever if not already in session state
if "retriever" not in st.session_state:
    # Load vectorstore using utility function
    vectorstore = load_vectorstore_from_utils()
    # Convert vectorstore to retriever and store in session state
    st.session_state.retriever = vectorstore.as_retriever()

# Initialize QA chain if not already in session state
if "qa_chain" not in st.session_state:
    # Get the LLM instance using utility function
    _llm = get_llm()
    # Create a RetrievalQA chain and store in session state
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=_llm, chain_type="stuff", retriever=st.session_state.retriever
    )

# Function to evaluate the consistency of the answer with the provided context
def evaluate_consistency(context, answer):
    # Define a prompt to check if the answer is supported by the context
    prompt = f"""
    Is this answer supported by the given context? Respond only 'Yes' or 'No'.
    Context: {context}
    Answer: {answer}
    """
    # Use the LLM to evaluate the prompt
    _llm = get_llm()
    return _llm.invoke(prompt).content.strip().lower()

# Streamlit app title
st.title("🔁 Corrective RAG")

# Input field for user query
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        # Retrieve relevant documents based on the query
        context_docs = st.session_state.retriever.invoke(query)
        # Combine the content of retrieved documents into a single string
        context_text = "\n".join([doc.page_content for doc in context_docs])
        
        # Get the answer from the QA chain
        result = st.session_state.qa_chain.invoke({"query": query})
        answer = result["result"]

        # Evaluate the consistency of the answer with the context
        if "no" in evaluate_consistency(context_text, answer):
            st.warning("⚠️ Initial answer might be inconsistent. Re-evaluating or re-querying (simplified for now)...")
            # For simplicity, we'll just use the same answer, but a real C-RAG might re-query or use a different prompt
            # result = st.session_state.qa_chain.invoke({"query": query}) # Option to re-query
            # answer = result["result"]

        # Display the final answer
        st.markdown("### Final Answer:")
        st.write(answer)