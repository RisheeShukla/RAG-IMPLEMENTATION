from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

loader=TextLoader("C:\\Users\\rishe\\OneDrive\\ドキュメント\\Desktop\\langchain_models\\Langchain-Runnables\\doc.txt",encoding="utf-8")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs=text_splitter.split_documents(documents)
embeddings=HuggingFaceEmbeddings()
vectorstore=FAISS.from_documents(docs,embeddings)
retriever=vectorstore.as_retriever()
query="Mohandas Karamchand Gandhi lived in South Africa for how many years?"

result=retriever.invoke(query)
print("Retrieved Documents:",result[0].page_content)
