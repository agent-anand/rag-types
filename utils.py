# Utility functions for the project
# Import necessary libraries
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Define constants for model names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3-8b-8192" # Or your preferred Groq model, e.g., "mixtral-8x7b-32768"

# Function to initialize and cache the LLM
@st.cache_resource
def get_llm():
    # Initialize the LLM with specified model name and temperature
    return ChatGroq(model=LLM_MODEL_NAME, temperature=0.7)

# Function to initialize and cache the embeddings model
@st.cache_resource
def get_embeddings_model():
    # Initialize the embeddings model with specified model name
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Function to load and cache the vectorstore
@st.cache_resource
def load_vectorstore_from_utils(): # Renamed to be specific
    # Load documents from the specified file
    _embeddings = get_embeddings_model()
    loader = TextLoader("data/documents.txt")
    docs = loader.load()
    # Split documents into chunks for processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    # Create a vectorstore from the split documents and embeddings
    vectorstore = FAISS.from_documents(split_docs, _embeddings)
    return vectorstore