from langchain_community.embeddings import HuggingFaceEmbeddings  # Import local embeddings class from LangChain
import os  # Import os for environment interactions

def get_embeddings_model():  # Define a function to initialize our embedding model
    """
    Initializes and returns the HuggingFace local embeddings model.
    This runs locally and does not require an API key.
    """
    # Define the specific model from HuggingFace. 'all-MiniLM-L6-v2' is efficient and high-performing for general tasks.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Specify the hardware device to use (standard CPUs are sufficient for this model size)
    model_kwargs = {'device': 'cpu'}
    
    # Set encoding parameters (normalize_embeddings=False is default for this specific model)
    encode_kwargs = {'normalize_embeddings': False}
    
    # Instantiate the HuggingFaceEmbeddings object with the defined parameters
    # This will download the model weights on the first execution
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings  # Return the ready-to-use embeddings model
