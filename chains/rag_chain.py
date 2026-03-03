from langchain_ollama import ChatOllama  # Import the Ollama chat model for local execution
from langchain.chains import ConversationalRetrievalChain  # Import the conversational retrieval chain class
from langchain.memory import ConversationBufferMemory  # Import the memory handler for chat history
from langchain.prompts import PromptTemplate  # Import the template class for custom prompt engineering
from dotenv import load_dotenv  # Import utility to load environment variables
import os  # Import os for file operations

# Initialize environment variables from the .env file
load_dotenv()

def get_rag_chain(vectorstore):  # Define a function to build the full conversational RAG chain
    """
    Sets up the ConversationalRetrievalChain with memory and Local Ollama LLM.
    Ensures a 100% free and private setup.
    """
    # Initialize the local LLM via Ollama. 
    # The 'llama3' model must be pulled on the user's system first.
    llm = ChatOllama(
        model="llama3",  # Specify the model name
        temperature=0,   # Set temperature to 0 for consistent, factual responses
    )
    
    # Initialize a memory buffer to store and pass conversational context between chat turns
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key used to store the dialogue history
        return_messages=True,       # Ensure messages are returned as actual objects, not just strings
        output_key="answer"         # Specify which output from the chain should be stored in memory
    )
    
    # Define a custom prompt template to enforce strict context adherence and source citation
    prompt_template = """
    Answer strictly from the provided context. If the answer is not present in the context, 
    respond that the document does not contain the information. 
    Do not use outside knowledge. 
    Always mention the source detail if available.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    # Create the actual PromptTemplate object with the defined logic and input variables
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # Assemble the ConversationalRetrievalChain using the defined LLM and retrieval system
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # The Large Language Model to use for generation
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # Convert the vector store into a retriever that fetches top 4 chunks
        memory=memory,  # The memory object initialized above
        return_source_documents=True,  # Ensure the chain returns the specific documents it retrieved
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},  # Pass the custom prompt to the underlying QA chain
        verbose=True  # Enable logging for easier debugging
    )
    
    return chain  # Return the fully orchestrated chain

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
