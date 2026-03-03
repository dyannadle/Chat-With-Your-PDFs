from langchain_community.document_loaders import PyPDFLoader  # Standard PDF loader
from langchain.schema import Document  # Document schema for LangChain
import pdfplumber  # Library for precise table and text extraction
import os  # Standard OS library
try:
    import pytesseract  # OCR library
    from pdf2image import convert_from_path  # PDF to image conversion for OCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def load_pdf_with_extras(file_path: str):  # Enhanced loader function
    """
    Loads a PDF using PyPDFLoader for text, pdfplumber for tables, 
    and pytesseract for OCR if text extraction fails.
    """
    documents = []  # List to hold extracted documents
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Try Standard Text Extraction (PyPDFLoader)
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Standard load failed for {file_path}, trying OCR: {e}")

    # 2. Table Extraction & Text Refinement (pdfplumber)
    # We use pdfplumber to get cleaner text from tables which standard loaders often mess up
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()  # Extract tables from the page
            if tables:
                table_text = f"\n[Table Data from Page {i+1}]:\n"
                for table in tables:
                    for row in table:
                        # Convert None values to empty strings and join cells with pipes
                        table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                
                # Append the table text as a new document chunk
                documents.append(Document(
                    page_content=table_text, 
                    metadata={"source": file_path, "page": i+1, "type": "table"}
                ))

    # 3. OCR Fallback (If no text was found and OCR libraries are installed)
    # This handles scanned PDFs as per requirement 4.1
    if OCR_AVAILABLE and (not documents or sum([len(d.page_content) for d in documents]) < 50):
        print(f"Running OCR on {file_path}...")
        images = convert_from_path(file_path) # Convert PDF pages to images
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image) # Run OCR on each image
            documents.append(Document(
                page_content=text, 
                metadata={"source": file_path, "page": i+1, "type": "ocr"}
            ))

    return documents  # Return all consolidated text/table chunks

def load_multiple_pdfs(file_paths: list):  # Consolidate multiple file paths
    """
    Iterates through file paths and extracts content with enhanced logic.
    """
    all_documents = []
    for path in file_paths:
        all_documents.extend(load_pdf_with_extras(path))
    return all_documents
