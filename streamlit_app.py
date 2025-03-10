
import os
import time
import base64
import shutil
import io
import openai
import sqlite3
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import inspect
from data import load_data
import streamlit as st
from streamlit_ace import st_ace

from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.responses.response_parser import ResponseParser

# ------------------------------------------------
# 1) Streamlit Page Config & Custom CSS
# ------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
/* Hide Streamlit default hamburger and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Customize the sidebar */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}

/* Title and header styles */
.big-title {
    font-size:2.0rem !important;
    font-weight:900 !important;
    color: #2B547E !important;
    margin-bottom: 0.3em;
}
.section-header {
    font-size:1.4rem !important;
    font-weight:700 !important;
    color: #003366 !important;
    margin-top:1em !important;
}

/* Subtle box styling */
.stTextArea, .stDataFrame, .st-code-block {
    border: 1px solid #dadada;
    border-radius: 4px;
    background-color: #fafafa;
    padding: 0.5em;
}

/* Buttons */
.css-1cpxqw2, .css-1q8dd3e, .stButton button {
    background-color: #006aff !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight:600 !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 2) OpenAI & Assistant Setup
# ------------------------------------------------
ASSISTANT_ID = "asst_HzB5u4pHtDOHQC6lGMIbg1Tk"
THREAD_ID = "thread_CTvS2U0BP4wJjna8rqXCLgCF"

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------
# 3) Session State
# ------------------------------------------------
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

# ------------------------------------------------
# 4) Data Loading
# ------------------------------------------------
df = load_data()

# ------------------------------------------------
# 5) Utility Functions
# ------------------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_chart_with_openai(image_path):
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

# ------------------------------------------------
# 6) Assistant API Functions
# ------------------------------------------------
def get_assistant_response(assistant_id, thread_id, user_input):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value
    except Exception as e:
        st.error(f"Error getting assistant response: {str(e)}")
        return "I'm sorry, but an error occurred while processing your request."

def get_avatar(role):
    if role == "user":
        return "https://www.themarysue.com/wp-content/uploads/2023/03/Tanjiro-Demon-Slayer.jpg"
    elif role == "assistant":
        return "https://ladygeekgirl.wordpress.com/wp-content/uploads/2015/10/mark-watney-matt-damon.jpg"
    else:
        return None

def display_chatbot():
    st.markdown('<h1 class="big-title">Assistant RAG Chatbot</h1>', unsafe_allow_html=True)
    st.write("**Focused on CC Fraud Information (10,000 Sample)**")

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything about the dataset!")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=get_avatar("user")):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=get_avatar("assistant")):
            message_placeholder = st.empty()
            full_response = get_assistant_response(ASSISTANT_ID, THREAD_ID, prompt)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ------------------------------------------------
# 7) PandasAI Callback Classes
# ------------------------------------------------
class StreamlitCallback(BaseCallback):
    def __init__(self, code_container, response_container) -> None:
        self.code_container = code_container
        self.response_container = response_container
        self.generated_code = ""

    def on_code(self, response: str):
        self.generated_code = response
        self.code_container.code(response, language="python")

    def get_generated_code(self):
        return self.generated_code

class StreamlitResponse(ResponseParser):
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

# ------------------------------------------------
# 8) App Layout - 5 Pages
# ------------------------------------------------
PAGE_OPTIONS = [
    "Dataset Overview",
    "Assistant Chat",
    "PandasAI Insights",
    "Code Editor",
    "Documentation"
]
page = st.sidebar.radio("Select a Page", PAGE_OPTIONS)

# -----------------------------------------------------
# PAGE 1: DATASET OVERVIEW
# -----------------------------------------------------
if page == "Dataset Overview":
    st.markdown('<h1 class="big-title">DATA FORGE</h1>', unsafe_allow_html=True)
    st.write("### Credit Card Fraud Detection")

    st.image("./images/fraud.png", width=300,
             caption="The world of digital transactions")

    st.markdown('<h2 class="big-title">Dataset Overview</h1>', unsafe_allow_html=True)
    st.write("### Data Preview")
    st.dataframe(df.head(5))

    st.markdown('<p class="section-header">Quick Statistics (df.describe())</p>', unsafe_allow_html=True)
    st.dataframe(df.describe())

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    info_str = buffer.getvalue()

    st.markdown('<p class="section-header">Dataset Info (df.info())</p>', unsafe_allow_html=True)
    st.text(info_str)

    st.info("Use the sidebar to navigate to other pages.")

# -----------------------------------------------------
# PAGE 2: ASSISTANT CHAT
# -----------------------------------------------------
elif page == "Assistant Chat":
    display_chatbot()

# -----------------------------------------------------
# PAGE 3: PANDASAI INSIGHTS
# -----------------------------------------------------
elif page == "PandasAI Insights":
    st.markdown('<h1 class="big-title">PandasAI Analysis & GPT-4 Vision</h1>', unsafe_allow_html=True)

    with st.expander("🔎 Dataframe Preview"):
        st.dataframe(df.head(5))

    with st.expander("💡 Sample Visualization Questions"):
        st.markdown("""
        - **Can you create a violin plot to show the variance in transaction amounts for fraud vs. non-fraud?**
        - **Plot a scatter chart of transaction amount vs. user account age, colored by fraud label.**
        - **Show a boxplot of transaction amounts grouped by transaction category (e.g., merchant type), highlighting fraud vs. non-fraud.**
        - **Create a line or bar chart showing how many fraudulent transactions occur by hour of the day, and compare it to non-fraudulent transactions. (Use TX_DURING_NIGHT or TX_DATETIME.)**
        - **Plot a scatter or box plot comparing TX_AMOUNT to CUSTOMER_ID_NB_TX_7DAY_WINDOW, colored by TX_FRAUD. Does high 7-day transaction frequency correlate with fraud?**
        - **Create a joint or pair plot of CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW and TERMINAL_ID_RISK_30DAY_WINDOW to see if there’s a pattern in fraud transactions.**
        - **Plot a histogram or bar chart of CUSTOMER_ID_NB_TX_30DAY_WINDOW for TX_FRAUD=1 vs. TX_FRAUD=0. Is there a user frequency range that stands out for fraud?**
        - **Use a heatmap or pair plot to see if TX_AMOUNT correlates with TERMINAL_ID_RISK_1DAY_WINDOW, TERMINAL_ID_RISK_7DAY_WINDOW, and TERMINAL_ID_RISK_30DAY_WINDOW.**
        - **Plot a line or scatter chart comparing fraud ratio to CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW binned into intervals. Does higher average spend lead to a higher or lower fraud ratio?**
        """)

    st.info("Please refresh the page for each new query")
    st.info("Please refresh the page for each new query.")

    st.markdown('<p class="section-header">Ask a Data Analysis Question</p>', unsafe_allow_html=True)
    st.session_state.pandasai_query = st.text_area(
        "Enter your analysis question here:",
        value=st.session_state.pandasai_query
    )

    if st.session_state.pandasai_query:
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

# -----------------------------------------------------
# PAGE 4: CODE EDITOR
# -----------------------------------------------------
elif page == "Code Editor":
    st.markdown('<h1 class="big-title">User Code Editor & Execution</h1>', unsafe_allow_html=True)

    if st.session_state.generated_code:
        st.markdown('<p class="section-header">AI-Generated Code (Editable)</p>', unsafe_allow_html=True)
        edited_code = st_ace(
            value=st.session_state.editor_code,
            language="python",
            theme="tomorrow",
            key="code_editor",
            height=750,
        )
        st.session_state.editor_code = edited_code

        run_button = st.button("🚀 Run Code")
        if run_button:
            try:
                exec_locals = {}
                exec(st.session_state.editor_code, {"df": df, "pd": pd, "plt": plt, "st": st, "sns": sns}, exec_locals)

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

        if st.session_state.user_code_result:
            result = st.session_state.user_code_result
            if result["type"] == "plot":
                st.markdown('<p class="section-header">User-Generated Plot</p>', unsafe_allow_html=True)
                st.image(result["value"])
            elif result["type"] == "dataframe":
                st.markdown('<p class="section-header">DataFrame Output</p>', unsafe_allow_html=True)
                st.dataframe(result["value"])
            elif result["type"] == "string":
                st.markdown(f"### 📌 AI Insight\n\n{result['value']}")
            elif result["type"] == "number":
                st.write(f"Result: {result['value']}")
    else:
        st.info("No generated code yet. Please go to 'PandasAI Insights' and enter a query first.")

# -----------------------------------------------------
# PAGE 5: DOCUMENTATION
# -----------------------------------------------------
else:
    st.markdown('<h1 class="big-title">Documentation</h1>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ## Overview
    1. **Dataset Overview**: A quick summary of the data (head, describe, info).
    2. **Assistant Chat**: A chatbot to retrieve data context from the OpenAI Playground (or vector store).
    3. **PandasAI Insights**: Uses PandasAI to generate code + GPT-4 Vision for chart interpretation.
    4. **Code Editor**: Modify and run the AI-generated code within Streamlit.
    5. **Documentation**: Shows the entire code with an explanation.

    ### Technical Details
    - **SQLite** data loading
    - **Session state** for preserving queries and code
    - **Chat** with custom avatars
    - **PandasAI** for data exploration
    - **GPT-4 Vision** for chart interpretation
    """)
