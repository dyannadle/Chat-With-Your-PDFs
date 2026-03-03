from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_embeddings_model():
    """
    Initializes and returns the OpenAI embeddings model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback or error based on project requirements. 
        # Spec asks for OpenAI/HuggingFace. Defaulting to OpenAI.
        pass 
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings
