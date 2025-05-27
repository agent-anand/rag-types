# 🧠 Retrieval-Augmented Generation (RAG) Playground

A Streamlit application designed to explore and demonstrate various Retrieval-Augmented Generation (RAG) techniques. This playground allows users to interact with different RAG implementations, understand their nuances, and see them in action with custom data.

## ✨ Features

*   Interactive Streamlit interface.
*   Multiple RAG variants showcased on separate pages.
*   Utilizes powerful LLMs via Groq API (e.g., Llama3, Mixtral).
*   Embeddings generated using HuggingFace Sentence Transformers.
*   Vector store powered by FAISS for efficient document retrieval.
*   Easy to configure with your own documents and API keys.

## 📚 RAG Variants Implemented

The application demonstrates the following RAG techniques:

1.  **🔹 Simple RAG**: A basic RAG pipeline that retrieves relevant documents and uses them as context for an LLM to generate an answer.
2.  **🔁 Corrective RAG (C-RAG)**: Aims to improve response quality by evaluating the initial answer's consistency with the retrieved context. If inconsistent, it signals the potential need for re-evaluation or re-querying (simplified in this demo).
3.  **🧠 Self-RAG (Pseudo)**: Simulates aspects of Self-RAG by performing a self-evaluation step. It checks if the generated answer is supported by the retrieved context and can attempt to regenerate if not.
4.  **🔀 Fusion RAG**: Explores query expansion by generating multiple sub-queries from the original user query. It then retrieves documents relevant to each sub-query. (Note: This demo focuses on the retrieval part; a full Fusion RAG would also involve re-ranking and synthesizing results).
5.  **🤖 Agentic RAG**: A conceptual page for agent-based RAG. The current implementation is a basic LLM call. A full Agentic RAG would involve an agent using tools (like a retriever tool) to dynamically decide how to best answer a query.
6.  **🔮 Speculative RAG**: Generates multiple candidate answers to a query and then scores these candidates based on their relevance to documents retrieved from the vector store. The highest-scoring candidate is selected as the final answer.

## 🛠️ Technologies Used

*   **Python 3.9+**
*   **Streamlit**: For building the interactive web application.
*   **Langchain**: Framework for developing applications powered by language models.
*   **Groq**: For fast LLM inference via API.
*   **FAISS**: For efficient similarity search and vector storage.
*   **HuggingFace Sentence Transformers**: For generating text embeddings.
*   **python-dotenv**: For managing environment variables.

## 🚀 Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd rag_type # Or your repository name
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    Ensure you have a `requirements.txt` file (see example below) and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the root directory of the project (e.g., `d:\AI Eagles\LanChain-Projects\rag_type\.env`).
    Add your Groq API key to this file:
    ```env
    GROQ_API_KEY="your_groq_api_key_here"
    ```
    You can obtain a Groq API key from GroqCloud.

5.  **Prepare your data**:
    *   Create a directory named `data` in the root of your project (e.g., `d:\AI Eagles\LanChain-Projects\rag_type\data`).
    *   Inside the `data` directory, create a file named `documents.txt`.
    *   Add the text content you want the RAG models to use as their knowledge base into `documents.txt`. The system will process this file to build its vector store.

## 🏃 Running the Application

Once the setup is complete, run the Streamlit application from the project's root directory using the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser. You can then navigate through the different RAG implementations using the sidebar.

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application entry point
├── pages/                  # Directory for different RAG model pages
│   ├── 1_Simple_RAG.py
│   ├── 2_Corrective_RAG.py
│   ├── 3_Self_RAG.py
│   ├── 4_Fusion_RAG.py
│   ├── 5_Agentic_RAG.py
│   └── 6_Speculative_RAG.py
├── utils.py                # Utility functions (LLM, embeddings, vectorstore loading)
├── data/
│   └── documents.txt       # Your custom knowledge base document
├── .env                    # Environment variables (e.g., API keys) - (You need to create this)
└── requirements.txt        # Python dependencies - (You need to create this or use the one provided)
```

## ⚙️ Configuration

*   **LLM Model**: The LLM model used (e.g., `llama3-8b-8192`) can be configured in `utils.py` via the `LLM_MODEL_NAME` constant.
*   **Embedding Model**: The embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) can be configured in `utils.py` via the `EMBEDDING_MODEL_NAME` constant.
*   **Data Source**: The RAG system uses `data/documents.txt` as its knowledge base. Modify this file with your own text data. The vector store is cached, so if you change `documents.txt`, you might need to clear Streamlit's cache or restart the app for changes to take full effect during development.

## 📝 Notes on Specific RAG Implementations

*   **Agentic RAG**: The current implementation in `pages/5_Agentic_RAG.py` is a simplified placeholder. A more robust Agentic RAG would involve defining specific tools for an agent (e.g., a retrieval tool that queries the vector store) and utilizing a Langchain agent executor to manage the interaction flow.
*   **Fusion RAG**: The current implementation focuses on the query expansion (generating sub-queries) and multi-document retrieval aspects. A complete Fusion RAG system would typically also incorporate steps for re-ranking the retrieved documents from all sub-queries and then synthesizing a final, coherent answer.

## 💡 Potential Future Enhancements

*   Implement more sophisticated re-ranking for Fusion RAG.
*   Develop a true Langchain agent with tools for Agentic RAG.
*   Add more evaluation metrics for comparing RAG techniques.
*   Allow users to upload their own documents through the UI.
*   Incorporate different vector stores or embedding models.

---

Happy RAG-ing!