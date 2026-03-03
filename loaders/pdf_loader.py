from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path: str):
    """
    Loads a PDF file and extracts its content.
    
    Note: For Scanned PDFs/OCR support as per spec:
    You can use UnstructuredPDFLoader from langchain_community.document_loaders:
    loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="ocr", ocr_languages="eng")
    This requires tesseract-ocr to be installed on the system.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Standard loader (handles most text-based PDFs and multi-column layouts)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_multiple_pdfs(file_paths: list):
    """
    Loads multiple PDF files and returns a combined list of documents.
    """
    all_documents = []
    for path in file_paths:
        all_documents.extend(load_pdf(path))
    return all_documents
