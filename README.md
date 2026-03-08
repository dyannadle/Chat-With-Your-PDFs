# 📄 Chat With Your PDFs

A production-ready, **100% free**, and **locally hosted** AI assistant that allows you to chat with your PDF documents conversationally. Built with **LangChain**, **Ollama**, **HuggingFace**, and **Streamlit**.

---

## 🚀 Key Features

### Core RAG Flow
- **Total Privacy**: Your documents and chats stay on your machine.
- **Zero Cost**: No API keys or subscriptions required.
- **Advanced RAG**: Uses Retrieval-Augmented Generation for accurate, document-based answers.
- **Conversational Memory**: Remembers your previous questions for a natural interaction.
- **Source Citations**: Clearly labels which document and page each answer came from.

### Advanced Enhancements (Phase 2)
- **Hybrid Search**: Combines semantic (vector) search with keyword (BM25) search for pinpoint accuracy.
- **OCR Support**: Automatically extract text from scanned PDFs using Tesseract.
- **Table Extraction**: Precise extraction of tabular data using `pdfplumber`.
- **Document Summarization**: Instantly generate a concise overview of your uploaded documents.
- **Chat Export to PDF**: Download your entire conversation transcript as a professional PDF.
- **Semantic Dashboard**: Explore the exact document chunks the AI retrieved for its answer.
- **Passage Highlighting**: Visual highlighting of retrieved text within the UI for better context.
- **File Limits**: Supports up to 200MB file uploads for processing large documents.

---

## 🛠 Tech Stack
- **Framework**: [LangChain](https://www.langchain.com/) for RAG orchestration.
- **LLM**: [Ollama](https://ollama.com/) (running Llama 3).
- **Embeddings**: [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Local).
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) for local persistence.
- **OCR**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
- **UI**: [Streamlit](https://streamlit.io/).

---

## 📦 Folder Structure
```text
chat_with_pdfs/
│
├── app.py                # Main Streamlit application UI
├── requirements.txt      # Python dependencies
│
├── loaders/
│   └── pdf_loader.py     # OCR & Table-aware PDF extraction
│
├── embeddings/
│   └── embedding_model.py # Local HuggingFace embedding setup
│
├── vectorstore/
│   └── vectordb.py       # FAISS database creation and loading
│
├── chains/
│   └── rag_chain.py       # Hybrid RAG chain & Summarization logic
│
├── utils/
│   ├── text_splitter.py   # Text chunking strategies
│   └── pdf_export.py     # Chat-to-PDF export utility
│
├── .streamlit/
│   └── config.toml       # Streamlit security & theme config
│
└── data/
    └── uploaded_docs/    # Directory for temporary PDF storage
```

---

## 📖 How to Run

### 1. Requirements
- **Python 3.10+**
- **Ollama** installed with `llama3` pulled.
- **Tesseract OCR** (Optional: only needed for scanned PDFs).

### 2. Install Ollama
Download and install [Ollama](https://ollama.com/). Then run:
```bash
ollama pull llama3
```

### 3. Setup the Project
```bash
# Clone the repository
cd "Chat With Your PDFs"

# Install Python packages
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run app.py
```
**Usage**: No password required for direct access.

---

## 📝 Code Annotations
Every single line of code in this project is exhaustively annotated to explain the underlying logic, from vector embeddings to hybrid retrieval.

---

## 📜 License
MIT License.
