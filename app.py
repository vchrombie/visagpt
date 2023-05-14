from datetime import datetime

import streamlit as st
from streamlit_chat import message

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title='🌍📝 VisaGPT: Your Immigration Assistant', layout="wide")

def generate_response(prompt):
    message = datetime.now().strftime("%H:%M:%S")
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
