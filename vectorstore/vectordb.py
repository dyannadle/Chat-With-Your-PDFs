from langchain_community.vectorstores import FAISS  # Import the FAISS vector database wrapper from LangChain
import os  # Import os for file path manipulations

def create_vectorstore(chunks, embeddings):  # Define a function to build the vector database from text chunks
    """
    Creates a FAISS vectorstore from text chunks.
    """
    # Use the FAISS.from_documents static method to convert text chunks into vector embeddings and index them
    # This process handles the heavy lifting of calculating vectors for every chunk
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore  # Return the populated vectorstore object

def save_vectorstore(vectorstore, path="data/vectorstore"):  # Define a function to save the vector index to disk
    """
    Saves the FAISS vectorstore locally.
    """
    # Serialize and save the vector store locally so it can be reloaded later without re-embedding
    vectorstore.save_local(path)

def load_vectorstore(path, embeddings):  # Define a function to reload an existing vector index
    """
    Loads a FAISS vectorstore from local storage.
    """
    if os.path.exists(path):  # Check if the saved index folder actually exists
        # Load the index from the local directory
        # allow_dangerous_deserialization is required for local FAISS loading using pickle internally
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        
    return None  # Return None if the folder is not found
