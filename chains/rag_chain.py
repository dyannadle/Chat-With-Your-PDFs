from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

from langchain.prompts import PromptTemplate

def get_rag_chain(vectorstore):
    """
    Sets up the ConversationalRetrievalChain with memory and Google Gemini LLM.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Custom Question Answering Prompt
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
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=True
    )
    
    return chain

def process_query(chain, query):
    """
    Processes a user query through the RAG chain and returns the response and sources.
    """
    result = chain({"question": query})
    answer = result["answer"]
    sources = []
    
    for doc in result["source_documents"]:
        source_info = {
            "page": doc.metadata.get("page", "N/A"),
            "source": os.path.basename(doc.metadata.get("source", "Unknown")),
            "content": doc.page_content[:200] + "..."
        }
        sources.append(source_info)
        
    return answer, sources
