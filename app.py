import streamlit as st
import os
from dotenv import load_dotenv
from loaders.pdf_loader import load_multiple_pdfs
from utils.text_splitter import get_text_chunks
from embeddings.embedding_model import get_embeddings_model
from vectorstore.vectordb import create_vectorstore, save_vectorstore, load_vectorstore
from chains.rag_chain import get_rag_chain, process_query

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="Chat With Your PDFs", page_icon="📄", layout="wide")
    
    # Custom CSS for a premium look
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #007bff;
            color: white;
        }
        .stChatInputContainer {
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("📄 Chat With Your PDFs")
    st.markdown("---")
    
    # Sidebar for PDF uploads
    with st.sidebar:
        st.header("Settings")
        uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    # Save files temporarily
                    temp_dir = "data/uploaded_docs"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # RAG Pipeline
                    raw_docs = load_multiple_pdfs(file_paths)
                    chunks = get_text_chunks(raw_docs)
                    embeddings = get_embeddings_model()
                    vectorstore = create_vectorstore(chunks, embeddings)
                    save_vectorstore(vectorstore)
                    
                    # Initialize session state for the chain
                    st.session_state.rag_chain = get_rag_chain(vectorstore)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")
        
        if st.button("Clear Chat Memory"):
            if "messages" in st.session_state:
                del st.session_state.messages
            if "rag_chain" in st.session_state:
                # Re-initialize to clear memory
                st.session_state.rag_chain = get_rag_chain(st.session_state.rag_chain.retriever.vectorstore)
            st.info("Chat history cleared.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for src in message["sources"]:
                        st.markdown(f"**Source:** {src['source']} (Page {src['page']})")
                        st.caption(src["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if "rag_chain" in st.session_state:
            with st.spinner("Thinking..."):
                answer, sources = process_query(st.session_state.rag_chain, prompt)
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        for src in sources:
                            st.markdown(f"**Source:** {src['source']} (Page {src['page']})")
                            st.caption(src["content"])
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
        else:
            st.info("Please upload and process documents first.")

if __name__ == "__main__":
    main()
