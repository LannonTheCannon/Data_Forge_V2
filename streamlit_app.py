import os
import time
import base64
import shutil
import openai
import sqlite3
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

import streamlit as st
from streamlit_ace import st_ace

from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.responses.response_parser import ResponseParser


# -----------------------------
# 1) Streamlit Setup + Page Config
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# 2) OpenAI & Assistant Setup
# -----------------------------
ASSISTANT_ID = "asst_HzB5u4pHtDOHQC6lGMIbg1Tk"  # Replace with your Assistant ID
THREAD_ID = "thread_CTvS2U0BP4wJjna8rqXCLgCF"   # Replace with your Thread ID

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# If you need the Beta endpoints (like client.beta.threads), ensure your openai lib is updated
# For demonstration, we'll assume the usage is correct.

# -----------------------------
# 3) Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "editor_code" not in st.session_state:
    st.session_state.editor_code = ""

if "chart_generated" not in st.session_state:
    st.session_state.chart_generated = False

if "chart_path" not in st.session_state:
    st.session_state.chart_path = ""

if "pandasai_query" not in st.session_state:
    st.session_state.pandasai_query = ""

if "gpt4_vision_text" not in st.session_state:
    st.session_state.gpt4_vision_text = None

if "user_code_result" not in st.session_state:
    st.session_state.user_code_result = None


# -----------------------------
# 4) Data Loading (SQLite Example)
# -----------------------------
def load_data():
    """
    Load your fraud dataset from SQLite, returning a DataFrame.
    Adjust if you have different table or row limits.
    """
    DB_FILE = "fraud_data.db"
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database file {DB_FILE} not found!")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM fraud_data LIMIT 1000;", conn)
    conn.close()
    return df

df = load_data()  # Load once at startup


# -----------------------------
# 5) Utility Functions
# -----------------------------
def encode_image(image_path):
    """Encodes an image to Base64 for OpenAI Vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_chart_with_openai(image_path):
    """Sends the generated chart to OpenAI GPT-4 Vision API and returns insights."""
    try:
        if not os.path.exists(image_path):
            return "⚠️ No chart found. Please generate a chart first."

        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a data visualization chart generated from a dataset related to credit card fraud detection. "
                                "Analyze the chart and provide insights by answering the following:\n\n"
                                "**1️⃣ Describe the chart:** What type of chart is this? What variables are on each axis?\n"
                                "**2️⃣ Identify trends & patterns:** Any notable trends, peaks, declines, clusters, or outliers?\n"
                                "**3️⃣ Interpret key findings:** What does it suggest about fraud detection?\n"
                                "**4️⃣ Provide actionable insights:** What recommendations or next steps would you suggest?\n\n"
                                "Be as detailed as possible and avoid assumptions beyond what the chart visually represents."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error analyzing chart: {e}"

# -----------------------------
# 6) Assistant API Functions
# -----------------------------
def get_assistant_response(assistant_id, thread_id, user_input):
    """Sends user_input to the Mark Watney Chatbot or any configured assistant."""
    try:
        # Post user message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )
        # Create a run
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        # Wait for the run to complete
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(1)
        # Retrieve the assistant's messages
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        # Return the latest assistant message
        return messages.data[0].content[0].text.value

    except Exception as e:
        st.error(f"Error getting assistant response: {str(e)}")
        return "I'm sorry, but an error occurred while processing your request."

def get_avatar(role):
    """Returns avatar URLs for user vs. assistant. Adjust as you like."""
    if role == "user":
        return "https://www.themarysue.com/wp-content/uploads/2023/03/Tanjiro-Demon-Slayer.jpg"
    elif role == "assistant":
        return "https://ladygeekgirl.wordpress.com/wp-content/uploads/2015/10/mark-watney-matt-damon.jpg"
    else:
        return None

def display_chatbot():
    """Displays the Mark Watney Chatbot with persistent messages."""
    st.title("Assistant RAG Chatbot with CC Fraud Information (1000)")

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything about the dataset!")
    if prompt:
        # User input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=get_avatar("user")):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant", avatar=get_avatar("assistant")):
            message_placeholder = st.empty()
            full_response = get_assistant_response(
                ASSISTANT_ID,
                THREAD_ID,
                prompt
            )
            message_placeholder.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# -----------------------------
# 7) PandasAI Callback Classes
# -----------------------------
class StreamlitCallback(BaseCallback):
    """Displays AI-generated code."""
    def __init__(self, code_container, response_container) -> None:
        self.code_container = code_container
        self.response_container = response_container
        self.generated_code = ""

    def on_code(self, response: str):
        """Displays AI-generated code"""
        self.generated_code = response
        self.code_container.code(response, language="python")

    def get_generated_code(self):
        return self.generated_code

class StreamlitResponse(ResponseParser):
    """
    Avoid double-loading charts. Just save to temp_chart.png if it's a plot.
    """
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        st.session_state.chart_generated = False

    def format_plot(self, result):
        chart_path = os.path.abspath("temp_chart.png")
        if isinstance(result["value"], str):
            existing_chart_path = os.path.abspath(result["value"])
            if existing_chart_path == chart_path:
                st.session_state.chart_generated = True
                st.session_state.chart_path = chart_path
            else:
                try:
                    shutil.copy(existing_chart_path, chart_path)
                    st.session_state.chart_generated = True
                    st.session_state.chart_path = chart_path
                except Exception as e:
                    st.error(f"⚠️ Error copying chart: {e}")
                    st.session_state.chart_generated = False
        elif isinstance(result["value"], plt.Figure):
            result["value"].savefig(chart_path)
            st.session_state.chart_generated = True
            st.session_state.chart_path = chart_path
        else:
            st.error("⚠️ Unexpected chart format returned.")
            st.session_state.chart_generated = False

    def format_other(self, result):
        st.markdown(f"### 📌 AI Insight\n\n{result['value']}")
        st.session_state.chart_generated = False


# -----------------------------
# 8) App Layout - Three Pages
# -----------------------------
page = st.sidebar.radio("Select a Page", ["Assistant Chat", "PandasAI Insights", "Code Editor"])

# ============================
# PAGE 1: Assistant Chat
# ============================
if page == "Assistant Chat":
    display_chatbot()

# ============================
# PAGE 2: PandasAI Insights
# ============================
elif page == "PandasAI Insights":
    st.title("PandasAI Analysis & GPT-4 Vision")

    with st.expander("🔎 Dataframe Preview"):
        st.dataframe(df.head(5))

    # Query for PandasAI
    st.session_state.pandasai_query = st.text_area(
        "🗣️ Ask a Data Analysis Question:",
        value=st.session_state.pandasai_query
    )

    if st.session_state.pandasai_query:
        # Create containers for code & response
        code_container = st.container()
        response_container = st.container()

        llm = PandasOpenAI(api_token=os.environ["OPENAI_API_KEY"])
        callback_handler = StreamlitCallback(code_container, response_container)

        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitResponse,
                "callback": callback_handler,
            },
        )

        answer = query_engine.chat(st.session_state.pandasai_query)
        generated_code = callback_handler.get_generated_code()
        st.session_state.generated_code = generated_code
        st.session_state.editor_code = generated_code

    # If a chart is generated, display + GPT-4 Vision
    if st.session_state.chart_generated:
        st.image(st.session_state.chart_path, caption="Chart from PandasAI")

        if st.session_state.gpt4_vision_text:
            st.markdown(f"### 📊 AI Chart Breakdown\n\n{st.session_state.gpt4_vision_text}")
            st.button("🧐 GPT-4 Vision Interpreter", disabled=True)
        else:
            gpt4_vision_button = st.button("🧐 GPT-4 Vision Interpreter")
            if gpt4_vision_button:
                st.session_state.gpt4_vision_text = analyze_chart_with_openai(st.session_state.chart_path)
                st.markdown(f"### 📊 AI Chart Breakdown\n\n{st.session_state.gpt4_vision_text}")
    else:
        st.button("🧐 GPT-4 Vision Interpreter", disabled=True)

    if not st.session_state.generated_code and not st.session_state.chart_generated:
        st.info("Enter a query above to generate a chart with PandasAI.")

# ============================
# PAGE 3: Code Editor
# ============================
else:
    st.title("User Code Editor & Execution")

    if st.session_state.generated_code:
        edited_code = st_ace(
            value=st.session_state.editor_code,
            language="python",
            theme="tomorrow",
            key="code_editor",
            height=750,
        )
        st.session_state.editor_code = edited_code

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")
        with col2:
            run_button = st.button("🚀 Run Code")

        if run_button:
            try:
                exec_locals = {}
                exec(
                    st.session_state.editor_code,
                    {"df": df, "pd": pd, "plt": plt, "st": st, "sns": sns},
                    exec_locals
                )

                if "analyze_data" in exec_locals:
                    analyze_func = exec_locals["analyze_data"]
                    params = inspect.signature(analyze_func).parameters

                    if "user_input" in params:
                        result = analyze_func([df], user_input="")
                    else:
                        result = analyze_func([df])

                    st.session_state.user_code_result = result

            except Exception as e:
                st.error(f"⚠️ Error running the code: {e}")

        # Show user-code output if it exists
        if st.session_state.user_code_result:
            result = st.session_state.user_code_result
            if result["type"] == "plot":
                st.image(result["value"])
            elif result["type"] == "dataframe":
                st.dataframe(result["value"])
            elif result["type"] == "string":
                st.markdown(f"### 📌 AI Insight\n\n{result['value']}")
            elif result["type"] == "number":
                st.write(f"Result: {result['value']}")
    else:
        st.info("No generated code yet. Please go to 'PandasAI Insights' and enter a query first.")
