# 📄 Chat With Your PDFs

A production-ready, **100% free**, and **locally hosted** AI assistant that allows you to chat with your PDF documents conversationally. Built with **LangChain**, **Ollama**, **HuggingFace**, and **Streamlit**.

---

## 🚀 Key Features
- **Total Privacy**: Your documents and chats stay on your machine.
- **Zero Cost**: No API keys or subscriptions required.
- **Advanced RAG**: Uses Retrieval-Augmented Generation for accurate, document-based answers.
- **Conversational Memory**: Remembers your previous questions for a natural interaction.
- **Source Citations**: Clearly labels which document and page each answer came from.
- **Modular Code**: Clean, annotated, and easy-to-extend architecture.

---

## 🛠 Tech Stack
- **Framework**: [LangChain](https://www.langchain.com/) for RAG orchestration.
- **LLM**: [Ollama](https://ollama.com/) (running Llama 3).
- **Embeddings**: [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Local).
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) for local persistence.
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
│   └── pdf_loader.py     # Logic for PDF text extraction
│
├── embeddings/
│   └── embedding_model.py # Local HuggingFace embedding setup
│
├── vectorstore/
│   └── vectordb.py       # FAISS database creation and loading
│
├── chains/
│   └── rag_chain.py       # Conversational RAG chain orchestration
│
├── utils/
│   └── text_splitter.py   # Text chunking strategies
│
└── data/
    └── uploaded_docs/    # Directory for temporary PDF storage
```

---

## 📖 How to Run

### 1. Requirements
Ensure you have **Python 3.10+** installed.

### 2. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/). Once installed, open your terminal and run:
```bash
ollama pull llama3
```

### 3. Setup the Project
Clone the repository and install the dependencies:
```bash
# Clone (if applicable) or navigate to the directory
cd "Chat With Your PDFs"

# Install Python packages
pip install -r requirements.txt
```

### 4. Launch the App
Run the Streamlit server:
```bash
streamlit run app.py
```
Your browser should automatically open to the app. Upload your PDFs, click **Process**, and start chatting!

---

## 📝 Code Annotations
Every file in this project is exhaustively annotated line-by-line to explain the logic and flow of the RAG system.

---

## 🤝 Contributing
Feel free to fork this project and add features like:
- Scanned PDF support (OCR).
- Chat exporting to PDF.
- Support for other file types (.docx, .txt).

---

## 📜 License
MIT License.
