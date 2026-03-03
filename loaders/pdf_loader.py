from langchain_community.document_loaders import PyPDFLoader  # Import the PDF loader class from LangChain community extensions
import os  # Import the standard os library for operating system interactions

def load_pdf(file_path: str):  # Define a function to load a single PDF file
    """
    Loads a PDF file and extracts its content.
    """
    if not os.path.exists(file_path):  # Check if the provided file path actually exists on the system
        raise FileNotFoundError(f"File not found: {file_path}")  # Raise an error if the file is missing
    
    # Initialize the PyPDFLoader with the given file path
    # This loader extracts text and metadata (like page numbers) from each page
    loader = PyPDFLoader(file_path)
    
    # Execute the load method to get a list of Document objects
    # Each Document object contains page_content (str) and metadata (dict)
    documents = loader.load()
    
    return documents  # Return the list of extracted documents

def load_multiple_pdfs(file_paths: list):  # Define a function to load multiple PDF files at once
    """
    Loads multiple PDF files and returns a combined list of documents.
    """
    all_documents = []  # Initialize an empty list to store all documents from all files
    
    for path in file_paths:  # Iterate through each file path provided in the list
        # Call the load_pdf function for the current path and extend the main list with its results
        all_documents.extend(load_pdf(path))
        
    return all_documents  # Return the consolidated list of Document objects from all PDFs
