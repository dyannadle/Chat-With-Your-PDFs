from langchain_community.vectorstores import FAISS
import os

def create_vectorstore(chunks, embeddings):
    """
    Creates a FAISS vectorstore from text chunks.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def save_vectorstore(vectorstore, path="data/vectorstore"):
    """
    Saves the FAISS vectorstore locally.
    """
    vectorstore.save_local(path)

def load_vectorstore(path, embeddings):
    """
    Loads a FAISS vectorstore from local storage.
    """
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None
