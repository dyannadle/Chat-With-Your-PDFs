# Chat With Your PDFs 📄

A production-ready RAG (Retrieval-Augmented Generation) assistant that allows you to chat with multiple PDF documents conversationally.

## Features
- **PDF Extraction**: Cleanly extracts text from single or multiple PDF files.
- **Smart Chunking**: Uses `RecursiveCharacterTextSplitter` to preserve semantic meaning.
- **Vector Search**: Local vector storage using FAISS for fast similarity search.
- **Conversational Memory**: Remembers past interactions for contextual follow-up questions.
- **Source Citations**: Provides references to the specific page and source for every answer.

## Tech Stack
- **Backend**: Python, LangChain, Ollama (Local LLM), HuggingFace (Local Embeddings), FAISS.
- **Frontend**: Streamlit.

## Setup Instructions

1. **Install Ollama**:
   - Download and install [Ollama](https://ollama.com/).
   - Pull the Llama 3 model:
     ```bash
     ollama pull llama3
     ```

2. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd "Chat With Your PDFs"
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Folder Structure
- `loaders/`: Logic for PDF text extraction.
- `utils/`: Text splitting and chunking strategies.
- `embeddings/`: Embedding model wrappers.
- `vectorstore/`: Vector database management.
- `chains/`: RAG orchestration and memory logic.
- `data/`: Local storage for uploaded documents and vector store.
