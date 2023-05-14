import os
from datetime import datetime

import streamlit as st
from streamlit_chat import message

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

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title='🌍📝 VisaGPT: Your Immigration Assistant', layout="wide")

llm = OpenAI(temperature=0)

# load the documents for us immigration
loader = DirectoryLoader('docs/us/',glob = "./*.pdf",loader_cls= PyPDFLoader)
# load the documents
documents = loader.load()
# chunk the documents into smaller pieces
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# split the database into text
texts = text_splitter.split_documents(documents)
# convert into embeddings
embeddings = OpenAIEmbeddings()

us = Chroma.from_documents(texts, embeddings, collection_name="us_citizen")
us = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=us.as_retriever())

# load the documents for canada immigration
loader = DirectoryLoader('docs/canada/',glob = "./*.pdf",loader_cls= PyPDFLoader)
# load the documents
documents = loader.load()
# chunk the documents into smaller pieces
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# split the database into text
texts = text_splitter.split_documents(documents)
# convert into embeddings
embeddings = OpenAIEmbeddings()

canada = Chroma.from_documents(texts, embeddings, collection_name="canada_citizen")
canada = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=canada.as_retriever())

# initialize the toolkit
tools = [
    Tool(
        name = "us immmigration run",
        func=us.run,
        description="answer questions about US immigration"
    ),
    Tool(
        name = "canada immmigration run",
        func=canada.run,
        description="answer questions about Canada immigration"
    ),
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    ai_prefix = "immigration_attorney",
    return_messages= True
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory = memory
)


def generate_response(prompt):
    message = agent.run(prompt)
    return message

with st.sidebar:
    st.title('🌍📝 VisaGPT')
    st.caption('Your Immigration Assistant')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](<https://streamlit.io/>)
    - [OpenAI](<https://openai.com/>)
    - [Langchain](<https://langchain.io/>)
    ''')
    st.write('Made with ❤️ at [HackGPT](<https://partiful.com/e/WjEpaOg8x6JwdKhHlNXL>)')
    st.write('Source code: [GitHub](<https://github.com/ek542/visagpt>)')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
