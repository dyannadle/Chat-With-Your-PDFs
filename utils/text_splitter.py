from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import the recursive text splitting logic from LangChain

def get_text_chunks(documents, chunk_size=1000, chunk_overlap=200):  # Define a function to split docs into chunks
    """
    Splits a list of documents into manageable chunks.
    """
    # Initialize the RecursiveCharacterTextSplitter.
    # It attempts to split on various characters (like newlines and spaces) to keep semantic blocks together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Set the maximum character count per chunk
        chunk_overlap=chunk_overlap,  # Set the number of characters to overlap between adjacent chunks for context preservation
        length_function=len,  # Define the function used to measure chunk length (standard character count)
        is_separator_regex=False,  # Specify that separators are literal strings, not regular expressions
    )
    
    # Split the provided list of Document objects into a larger list of smaller Document chunks
    chunks = text_splitter.split_documents(documents)
    
    return chunks  # Return the list of text chunks
