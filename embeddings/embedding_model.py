from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings_model():
    """
    Initializes and returns the HuggingFace local embeddings model.
    This runs locally and does not require an API key.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
