import streamlit as st  # Import Streamlit for the web application interface
import os  # Import os for file and directory operations
from dotenv import load_dotenv  # Import load_dotenv to read environment variables from .env file
from loaders.pdf_loader import load_multiple_pdfs  # Import custom PDF loading function
from utils.text_splitter import get_text_chunks  # Import custom text splitting utility
from embeddings.embedding_model import get_embeddings_model  # Import custom embedding model initializer
from vectorstore.vectordb import create_vectorstore, save_vectorstore, load_vectorstore  # Import vector store management functions
from chains.rag_chain import get_rag_chain, process_query  # Import RAG orchestration and query processing functions

# Load environment variables (like API keys, if any were used)
load_dotenv()

def main():  # Define the main function for the Streamlit app
    st.set_page_config(page_title="Chat With Your PDFs", page_icon="📄", layout="wide")  # Configure the browser tab title, icon, and layout
    
    # Inject custom CSS for a premium look and feel
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;  /* Set a light grey background for the main area */
        }
        .stButton>button {
            width: 100%;  /* Make buttons span the full width of their container */
            border-radius: 5px;  /* Round the button corners */
            height: 3em;  /* Set a fixed height for buttons */
            background-color: #007bff;  /* Use a professional blue color for buttons */
            color: white;  /* Use white text for buttons */
        }
        .stChatInputContainer {
            padding: 20px;  /* Add padding around the chat input area */
        }
        </style>
        """, unsafe_allow_html=True)  # Allow HTML in markdown to apply the CSS
    
    st.title("📄 Chat With Your PDFs")  # Display the main application title
    st.markdown("---")  # Add a horizontal divider line
    
    # Sidebar section for configurations and file uploads
    with st.sidebar:
        st.header("Settings")  # Display a header in the sidebar
        uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)  # File uploader for multiple PDF files
        
        if st.button("Process Documents"):  # Trigger processing when this button is clicked
            if uploaded_files:  # Check if any files were actually uploaded
                with st.spinner("Processing documents..."):  # Show a loading spinner during execution
                    # Define a directory to store uploaded files temporarily
                    temp_dir = "data/uploaded_docs"
                    if not os.path.exists(temp_dir):  # If the directory doesn't exist
                        os.makedirs(temp_dir)  # Create the directory
                    
                    file_paths = []  # Initialize a list to hold the saved file paths
                    for uploaded_file in uploaded_files:  # Iterate through each uploaded file
                        file_path = os.path.join(temp_dir, uploaded_file.name)  # Generate the full file path
                        with open(file_path, "wb") as f:  # Open the path for writing in binary mode
                            f.write(uploaded_file.getbuffer())  # Save the file content to disk
                        file_paths.append(file_path)  # Add the path to our list
                    
                    # Core RAG Pipeline execution
                    raw_docs = load_multiple_pdfs(file_paths)  # Step 1: Extract text and metadata from PDFs
                    chunks = get_text_chunks(raw_docs)  # Step 2: Split the text into smaller, manageable chunks
                    embeddings = get_embeddings_model()  # Step 3: Initialize the local embedding model
                    vectorstore = create_vectorstore(chunks, embeddings)  # Step 4: Create a FAISS vector store from chunks
                    save_vectorstore(vectorstore)  # Step 5: Persist the vector store to local storage
                    
                    # Step 6: Initialize the conversational retrieval chain and save it to session state
                    st.session_state.rag_chain = get_rag_chain(vectorstore)
                    st.success("Documents processed successfully!")  # Show a success message
            else:
                st.warning("Please upload at least one PDF.")  # Warn if the button was clicked without files
        
        if st.button("Clear Chat Memory"):  # Trigger chat history clearing
            if "messages" in st.session_state:  # Check if messages exist in session state
                del st.session_state.messages  # Delete the message history
            if "rag_chain" in st.session_state:  # If the RAG chain exists
                # Re-initialize the chain to reset its internal memory while keeping the retriever
                st.session_state.rag_chain = get_rag_chain(st.session_state.rag_chain.retriever.vectorstore)
            st.info("Chat history cleared.")  # Notify the user that history is reset

    # Initialize chat history in session state if it doesn't already exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Iterate through chat history and display each message on the UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # Create a chat bubble based on the role (user or assistant)
            st.markdown(message["content"])  # Display the message text
            if "sources" in message:  # If the message has source citations
                with st.expander("View Sources"):  # Create a collapsible expander for sources
                    for src in message["sources"]:  # Iterate through each source
                        st.markdown(f"**Source:** {src['source']} (Page {src['page']})")  # Display source file and page
                        st.caption(src["content"])  # Display a snippet of the source content

    # Capture user input from the chat box
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display the user's message immediately in the UI
        st.chat_message("user").markdown(prompt)
        # Append the user's message to the session history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if "rag_chain" in st.session_state:  # Check if documents have been processed
            with st.spinner("Thinking..."):  # Show a "Thinking..." spinner while the AI processes
                # Process the query through the RAG chain to get the answer and sources
                answer, sources = process_query(st.session_state.rag_chain, prompt)
                
                # Display the assistant's response in a new chat bubble
                with st.chat_message("assistant"):
                    st.markdown(answer)  # Display the AI's answer
                    with st.expander("View Sources"):  # Show sources in an expander
                        for src in sources:  # Loop through sources
                            st.markdown(f"**Source:** {src['source']} (Page {src['page']})")  # Cite source
                            st.caption(src["content"])  # Show snippet
                
                # Add the assistant's response and sources to the chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
        else:
            st.info("Please upload and process documents first.")  # Inform user if they skip processing

if __name__ == "__main__":  # Ensure the main function only runs if the script is executed directly
    main()
