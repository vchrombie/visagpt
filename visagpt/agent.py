import os

import streamlit as st

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents import AgentType
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.chains import RetrievalQA

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

@st.cache_data
def load_documents(_loader):
    return _loader.load()

@st.cache_data
def split_documents(_text_splitter, _documents):
    return _text_splitter.split_documents(_documents)

@st.cache_resource
def create_chroma_from_documents(_texts, _embeddings, _collection_name):
    return Chroma.from_documents(_texts, _embeddings, _collection_name=_collection_name)

@st.cache_resource
def load_llm():
    return OpenAI(temperature=0)

@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings()


llm = load_llm()
embeddings = load_embeddings()

@st.cache_resource
def load_and_process_documents(path):
    documents = load_documents(DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader))
    texts = split_documents(CharacterTextSplitter(chunk_size=200, chunk_overlap=0), documents)
    chroma = create_chroma_from_documents(texts, embeddings, "us_citizen")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=chroma.as_retriever())

us = load_and_process_documents('docs/us/')
canada = load_and_process_documents('docs/canada/')

@st.cache_resource()
def load_agent():
    tools = [
        Tool(
            name = "us immigration run",
            func=us.run,
            description="answer questions about us immigration"
        ),
        Tool(
            name = "canada immigration run",
            func=canada.run,
            description="answer questions about canada immigration"
        ),
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        ai_prefix = "immigration_attorney",
        return_messages= True
    )

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory = memory
    )

agent = load_agent()
