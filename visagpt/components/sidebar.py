import streamlit as st

from .faq import faq


def sidebar():
    with st.sidebar:
        st.title("🌍📝 VisaGPT")
        st.caption("🚀 Navigating Immigration with AI 🧠")
        st.markdown("""
        LLM-powered chatbot built using:
        - [Streamlit](<https://streamlit.io/>)
        - [OpenAI](<https://openai.com/>)
        - [Langchain](<https://langchain.io/>)
        """)
        st.write("""
            Made with ❤️ at
             [HackGPT](<https://partiful.com/e/WjEpaOg8x6JwdKhHlNXL>)
            """)
        st.write("Source code: [GitHub](<https://github.com/ek542/visagpt>)")
        st.markdown("""
        Built by:
        - [Eesha Khanna](<https://github.com/ek542>)
        - [Deepa Korani](<https://github.com/deepakorani>)
        - [Venu Vardhan Reddy Tekula](<https://github.com/vchrombie>)
        """)
        st.markdown("---")

        faq()
