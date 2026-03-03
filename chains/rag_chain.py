from langchain_ollama import ChatOllama  # Import the Ollama chat model for local execution
from langchain.chains import ConversationalRetrievalChain  # Import the conversational retrieval chain class
from langchain.memory import ConversationBufferMemory  # Import the memory handler for chat history
from langchain.prompts import PromptTemplate  # Import the template class for custom prompt engineering
from langchain.retrievers import BM25Retriever, EnsembleRetriever  # Import retrievers for hybrid search
from dotenv import load_dotenv  # Import utility to load environment variables
import os  # Import os for file operations

# Initialize environment variables from the .env file
load_dotenv()

def get_hybrid_retriever(chunks, vectorstore):  # Define a function to create a hybrid retriever
    """
    Combines BM25 (keyword) and FAISS (vector) retrievers for better accuracy.
    """
    # Initialize the keyword-based BM25 retriever using the text chunks
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2  # Retrieve top 2 chunks via keyword search
    
    # Initialize the vector-based FAISS retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 chunks via vector search
    
    # Combine them using an EnsembleRetriever with specific weightings (50/50 balance)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def get_rag_chain(vectorstore, chunks=None):  # Update the chain to support optional hybrid retrieval
    """
    Sets up the ConversationalRetrievalChain with memory and LLM.
    Supports Hybrid Search if chunks are provided.
    """
    # Initialize the local LLM via Ollama
    llm = ChatOllama(model="llama3", temperature=0)
    
    # Initialize conversational memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Custom QA Prompt
    prompt_template = """
    Answer strictly from the provided context. If the answer is not present in the context, 
    respond that the document does not contain the information. 
    Context:
    {context}
    Question: 
    {question}
    Answer:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Determine which retriever to use (Hybrid vs standard Vector)
    if chunks:
        retriever = get_hybrid_retriever(chunks, vectorstore)
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=True
    )
    
    return chain

def summarize_documents(documents):  # Define a function to generate a summary of the loaded documents
    """
    Generates a concise summary of the provided list of Document objects.
    """
    # Initialize a local LLM specifically for summarization tasks
    llm = ChatOllama(model="llama3", temperature=0)
    
    # Combine the content of the first few documents/pages to avoid context window issues
    # We take the first 5000 characters as a representative sample for the summary
    combined_text = "\n".join([doc.page_content for doc in documents[:5]]) 
    
    # Define a prompt specifically for summarization
    summary_prompt = f"""
    Please provide a concise and professional summary of the following document content. 
    Focus on the main topics, purpose, and key takeaways.
    
    Content:
    {combined_text[:5000]}
    
    Summary:
    """
    
    # Generate the summary using the LLM's invoke method
    response = llm.invoke(summary_prompt)
    
    # Extract and return the text content of the summary
    return response.content


def process_query(chain, query):  # Define a helper function to run queries through the chain
    """
    Processes a user query through the RAG chain and returns the response and sources.
    """
    # Execute the chain with the user's question
    result = chain({"question": query})
    
    # Extract the AI's answer from the result dictionary
    answer = result["answer"]
    
    # Initialize a list to hold formatted source information
    sources = []
    
    # Process each source document retrieved during the query execution
    for doc in result["source_documents"]:
        source_info = {
            "page": doc.metadata.get("page", "N/A"),  # Retrieve page number from metadata, or 'N/A' if missing
            "source": os.path.basename(doc.metadata.get("source", "Unknown")),  # Extract file name from the source path
            "content": doc.page_content[:200] + "..."  # Capture a short snippet of the actual text
        }
        # Add the formatted source to our list
        sources.append(source_info)
        
    return answer, sources  # Return both the generated answer and the list of sources
