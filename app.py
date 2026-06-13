from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACE_API_TOKEN')
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACEHUB_API_TOKEN
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    task="text-generation",
   huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=2,
    max_new_tokens=100
)
model=ChatHuggingFace(llm=llm)

st.title("RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx"]
)

query = st.text_input("Enter Query")

import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces/newlines
    return text.strip()

def fetch_info():
    embeddings =HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
)
    temp_path = uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = os.path.splitext(temp_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif ext == ".txt":
        loader = TextLoader(temp_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(temp_path)
    else:
        st.error("Unsupported file type")
        return

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    results = retriever.invoke(query)

    with st.chat_message("assistant"):
        text=""
        for i, doc in enumerate(results, 1):
            text+= clean_text(doc.page_content)
        st.header('Data fetched by RAG') 
        st.chat_message("assistant").write(text)
        st.header('Data modified by LLM')
        result=model.invoke(f"write the given data {text} in a clean way")
        st.chat_message("assistant").write(result.content)
    



if uploaded_file and query:
    st.button("Search", on_click=fetch_info)
