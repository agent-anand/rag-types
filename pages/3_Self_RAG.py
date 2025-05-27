# Streamlit page for Self-RAG functionality

# Import necessary libraries
import streamlit as st
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from utils import get_llm, load_vectorstore_from_utils
load_dotenv()

# Initialize chat history if not already in session state
if "chat_history_self_rag" not in st.session_state:
    st.session_state.chat_history_self_rag = []

# Streamlit app title
st.title("🧠 Pseudo Self-RAG")
st.markdown("Simulates Self-RAG-style self-evaluation using a pre-loaded text file (`data.txt`).")

# Load vectorstore and retriever
vectorstore = load_vectorstore_from_utils()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=get_llm(),
    chain_type="stuff",
    retriever=retriever
)

# Display chat history
for message in st.session_state.chat_history_self_rag:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about the content in data.txt...")
if user_input:
    # Append user input to chat history
    st.session_state.chat_history_self_rag.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating response..."):
            # Retrieve and generate response using QA chain
            result = qa_chain.invoke({"query": user_input})
            raw_answer = result["result"]
            context_docs = retriever.invoke(user_input) # Get context for evaluation
            context = "\n".join([doc.page_content for doc in context_docs])

            # Evaluate if answer is supported by context
            eval_prompt = f"""
            Is the following answer supported by the provided context? Answer only 'Yes' or 'No'.
            
            Context: {context}
            Answer: {raw_answer}
            """
            _llm = get_llm()
            evaluation = _llm.invoke(eval_prompt).content.strip().lower()

            if "no" in evaluation:
                st.markdown("⚠️ Inconsistent result. Regenerating with more focus...")
                result = qa_chain.invoke({"query": user_input}) # Re-invoke with proper input
                refined_answer = result["result"]
            else:
                refined_answer = raw_answer

            # Display the refined answer
            st.markdown(refined_answer)
            st.session_state.chat_history_self_rag.append({
                "role": "assistant", "content": refined_answer
            })

