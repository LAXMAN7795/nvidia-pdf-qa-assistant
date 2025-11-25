import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# llm
llm = ChatNVIDIA(model="meta/llama-3.2-1b-instruct", temperature=0.2, top_p=0.7)

# embeddings
def vectore_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
    return st.session_state.vectors

st.title("PDF Question Answering with NVIDIA LLMs")

# prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful AI assistant. Use the following context to answer the question.\n\n"
    "Context: {context}\n\n"
    "Question: {input}\n\n"
    "Answer the question in a concise manner."
)

prompt1 = st.text_input("Ask a question about the PDFs:")

if st.button("Document Embedding"):
    vectore_embedding()
    st.success("Document embedding completed!")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    response = retriever_chain.invoke({"input": prompt1})
    st.write("Answer:")
    st.write(response['answer'])


    
