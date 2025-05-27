# Entry point for the application

# Import necessary libraries
import streamlit as st
import sys

# Prevent Streamlit from watching torch.classes (workaround for faiss/torch bug)
sys.modules["torch.classes"] = None

# Set up the Streamlit page configuration
st.set_page_config(page_title="📚 Multi-Page RAG App", layout="centered")

# App title and description
st.title("🧠 Retrieval-Augmented Generation (RAG) Playground")
st.markdown("Select a RAG variant from the sidebar to try out!")

# Sidebar for model selection
st.sidebar.title("🔍 RAG Models")
st.sidebar.markdown("Choose one of the RAG implementations below:")

# Links to different RAG model pages
st.sidebar.page_link("pages/1_Simple_RAG.py", label="🔹 Simple RAG")
st.sidebar.page_link("pages/2_Corrective_RAG.py", label="🔁 Corrective RAG")
st.sidebar.page_link("pages/3_Self_RAG.py", label="🧠 Self RAG") # Or "Pseudo Self-RAG"
st.sidebar.page_link("pages/4_Fusion_RAG.py", label="🔀 Fusion RAG")
st.sidebar.page_link("pages/5_Agentic_RAG.py", label="🤖 Agentic RAG")
st.sidebar.page_link("pages/6_Speculative_RAG.py", label="🔮 Speculative RAG")

# Separator
st.markdown("---")

# Running instructions
st.markdown("### How to Run:")
st.code("streamlit run app.py", language="bash")