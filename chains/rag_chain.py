from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

def get_rag_chain(vectorstore):
    """
    Sets up the ConversationalRetrievalChain with memory and LLM.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
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
