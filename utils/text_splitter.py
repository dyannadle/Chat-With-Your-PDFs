from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits a list of documents into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
