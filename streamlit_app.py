import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
from data import load_data

class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

st.write("# Chat with Credit Card Fraud Dataset 🦙")

# Debugging: Check if database loads
st.write("Loading data from SQLite...")
df = load_data()
st.write(f"✅ Data Loaded: {df.shape}")

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(5))

query = st.text_area("🗣️ Chat with Dataframe")
container = st.container()

if query:
    st.write(f"Processing query: `{query}`")

    try:
        llm = OpenAI(api_token=st.secrets["OPENAI_API_KEY"])
        st.write("✅ OpenAI API Key loaded successfully!")

        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitResponse,
                "callback": StreamlitCallback(container),
            },
        )

        st.write("Sending query to SmartDataframe...")
        answer = query_engine.chat(query)
        st.write(f"✅ Response: {answer}")

    except Exception as e:
        st.error(f"⚠️ Error processing query: {e}")
