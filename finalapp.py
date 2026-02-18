import streamlit as st
import os
import time
from dotenv import load_dotenv

# NVIDIA & FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

# Core 2026 LCEL Components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# FIX: Defensive check for API Key to prevent TypeError
nv_api_key = os.getenv("NVIDIA_API_KEY")
if nv_api_key:
    os.environ['NVIDIA_API_KEY'] = nv_api_key
else:
    st.error("NVIDIA_API_KEY not found in .env file!")
    st.stop()

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        
        # In 2026, we process only the necessary subset for demo speed
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB Is Ready")

st.title("Nvidia NIM Demo 2026")

# Using the most stable 2026 model ID
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please embed documents first!")
    else:
        # MODERN 2026 LCEL RAG PATTERN (Replaces create_retrieval_chain)
        retriever = st.session_state.vectors.as_retriever()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the chain: Retrieve -> Format -> Prompt -> LLM -> Parse
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        start = time.process_time()
        # In LCEL, we invoke the chain directly
        response_text = rag_chain.invoke(prompt1)
        
        st.write(f"**Response time:** {time.process_time() - start:.2f}s")
        st.write(response_text)

        # To show sources, we retrieve them separately in 2026
        with st.expander("Document Similarity Search"):
            relevant_docs = retriever.invoke(prompt1)
            for i, doc in enumerate(relevant_docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")