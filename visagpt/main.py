import streamlit as st
from streamlit_chat import message

from components.sidebar import sidebar

st.set_page_config(page_title='🌍📝 VisaGPT 🚀🧠', layout="wide")

from agent import agent

def generate_response(prompt):
    try:
        message = agent.run(prompt)
    except ValueError:
        message = "Sorry, I don't understand. Please try again."

    return message

sidebar()

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
