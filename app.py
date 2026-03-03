import streamlit as st  # Import Streamlit for the web application interface
import os  # Import os for file and directory operations
from dotenv import load_dotenv, set_key  # Import dotenv utilities for persistent config
from loaders.pdf_loader import load_multiple_pdfs  # Import custom PDF loading function
from utils.text_splitter import get_text_chunks  # Import custom text splitting utility
from embeddings.embedding_model import get_embeddings_model  # Import custom embedding model initializer
from vectorstore.vectordb import create_vectorstore, save_vectorstore, load_vectorstore  # Import vector store management functions
from chains.rag_chain import get_rag_chain, process_query, summarize_documents  # Import RAG and summarization functions
from utils.pdf_export import export_chat_to_pdf  # Import the chat export utility

# Load environment variables (like API keys, if any were used)
load_dotenv()

def check_password():  # Define a simple authentication function for local privacy
    """Returns `True` if the user had the correct password."""
    def password_entered():
        # Retrieve the current password from environment variables, defaulting to 'admin123'
        correct_password = os.getenv("APP_PASSWORD", "admin123")
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

def main():  # Define the main function for the Streamlit app
    if not check_password():  # Only proceed if the password is correct
        st.stop()
        
    st.set_page_config(page_title="Chat With Your PDFs", page_icon="📄", layout="wide")
    
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
            padding: 20px;
        }
        .highlight {
            background-color: #fff3cd; /* Light yellow background for highlighting */
            border-left: 5px solid #ffc107; /* Thicker yellow left border */
            padding: 10px;
            border-radius: 4px;
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
                    st.session_state.rag_chain = get_rag_chain(vectorstore, chunks=chunks)
                    st.session_state.raw_docs = raw_docs  # Save raw docs for summarization
                    st.success("Documents processed successfully!")  # Show a success message
            else:
                st.warning("Please upload at least one PDF.")  # Warn if the button was clicked without files
        
        if st.button("Summarize Documents"):  # Trigger document summarization
            if "raw_docs" in st.session_state:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = summarize_documents(st.session_state.raw_docs)
                    st.success("Summary generated!")
            else:
                st.warning("Please process documents first.")

        if st.button("Export Chat to PDF"):  # Trigger chat export
            if "messages" in st.session_state and st.session_state.messages:
                with st.spinner("Generating PDF..."):
                    export_path = export_chat_to_pdf(st.session_state.messages)
                    with open(export_path, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name="chat_export.pdf",
                            mime="application/pdf"
                        )
            else:
                st.warning("No chat history to export.")
        
        if st.button("Clear Chat Memory"):  # Trigger chat history clearing
            if "messages" in st.session_state:  # Check if messages exist in session state
                del st.session_state.messages  # Delete the message history
            if "rag_chain" in st.session_state:  # If the RAG chain exists
                # Re-initialize the chain to reset its internal memory while keeping the retriever
                st.session_state.rag_chain = get_rag_chain(st.session_state.rag_chain.retriever.vectorstore)
            st.info("Chat history cleared.")  # Notify the user that history is reset
            
        st.markdown("---")
        st.header("🔐 Security Settings")  # Add a section for security
        new_password = st.text_input("Change App Password", type="password")
        if st.button("Update Password"):
            if new_password:
                # Persistently update the .env file with the new password
                set_key(".env", "APP_PASSWORD", new_password)
                st.success("Password updated! Please restart the app for changes to take effect.")
            else:
                st.warning("Please enter a valid password.")

    # Display document summary if available
    if "summary" in st.session_state:
        with st.expander("📄 Document Summary", expanded=True):
            st.write(st.session_state.summary)

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
                            st.markdown(f'<div class="highlight">{src["content"]}</div>', unsafe_allow_html=True) # Highlighted content
                
                # Add the assistant's response and sources to the chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
                # Save the most recent retrieval results for the Semantic Dashboard
                st.session_state.last_retrieval = sources
        else:
            st.info("Please upload and process documents first.")  # Inform user if they skip processing

    # ---------------------------------------------------------
    # 🔍 Semantic Search Dashboard (Requirement 7.0/4.4)
    # ---------------------------------------------------------
    if "last_retrieval" in st.session_state:
        st.markdown("---")
        st.subheader("🔍 Semantic Search Dashboard")
        st.markdown("Explore the specific document chunks the AI used to generate the last answer.")
        
        cols = st.columns(2)  # Create two columns for the dashboard
        for i, src in enumerate(st.session_state.last_retrieval):
            with cols[i % 2]:  # Alternating columns
                st.info(f"📍 **Chunk {i+1}** | {src['source']} (Page {src['page']})")
                st.caption(f"_{src['content']}_")
                st.markdown("---")

if __name__ == "__main__":  # Ensure the main function only runs if the script is executed directly
    main()
