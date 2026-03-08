import streamlit as st
import sys
import os

st.title("Debug Environment")
st.write(f"Python version: {sys.version}")
st.write(f"Executable: {sys.executable}")
st.write(f"Current Path: {os.getcwd()}")

try:
    import langchain
    st.write(f"LangChain version: {langchain.__version__}")
    st.write(f"LangChain file: {langchain.__file__}")
except Exception as e:
    st.error(f"LangChain import failed: {e}")

st.write("Checking for ConversationalRetrievalChain...")
try:
    from langchain.chains import ConversationalRetrievalChain
    st.success("Found in langchain.chains")
except Exception as e:
    st.error(f"Failed in langchain.chains: {e}")

try:
    from langchain_community.chains import ConversationalRetrievalChain
    st.success("Found in langchain_community.chains")
except Exception as e:
    st.error(f"Failed in langchain_community.chains: {e}")

try:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    st.success("Found in langchain.chains.conversational_retrieval.base")
except Exception as e:
    st.error(f"Failed in langchain.chains.conversational_retrieval.base: {e}")

import pkgutil
import langchain
import langchain_community

st.write("Submodules of langchain_community.chains:")
try:
    import langchain_community.chains as chains
    st.write([name for _, name, _ in pkgutil.iter_modules(chains.__path__)])
except Exception as e:
    st.error(f"Error listing langchain_community.chains submodules: {e}")

st.write("Sys Path:")
st.write(sys.path)
