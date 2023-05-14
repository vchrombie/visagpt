import streamlit as st

from .faq import faq

def sidebar():
    with st.sidebar:
        st.title('🌍📝 VisaGPT')
        st.caption('🚀 Navigating Immigration with AI 🧠')
        st.markdown('''
        An LLM-powered chatbot built using:
        - [Streamlit](<https://streamlit.io/>)
        - [OpenAI](<https://openai.com/>)
        - [Langchain](<https://langchain.io/>)
        ''')
        st.markdown('''
        > Sample Questions\
        
        > Q. What are the documents required for study visa in the US?\
        
        > Q. How long does it take to fill the Form DS-160?
        ''')
        st.write('Made with ❤️ at [HackGPT](<https://partiful.com/e/WjEpaOg8x6JwdKhHlNXL>)')
        st.write('Source code: [GitHub](<https://github.com/ek542/visagpt>)')
        st.markdown('''
        Built by:
        - [Eesha Khanna](<https://github.com/ek542>)
        - [Deepa Korani](<https://github.com/deepakorani>)
        - [Venu Vardhan Reddy Tekula](<https://github.com/vchrombie>)
        ''')
        st.markdown('Scroll more for FAQs!')
        st.markdown('---')

        faq()
