import os
import base64
import shutil
import openai
import streamlit as st
from streamlit_ace import st_ace
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.responses.response_parser import ResponseParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

# Enable Wide Mode
st.set_page_config(layout="wide")

# -----------------------
# 🔹 OpenAI Setup
# -----------------------
client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# -----------------------
# 🔹 Global Session State
# -----------------------
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "editor_code" not in st.session_state:
    st.session_state.editor_code = ""

if "chart_generated" not in st.session_state:
    st.session_state.chart_generated = False  # Track if PandasAI created a chart

if "chart_path" not in st.session_state:
    st.session_state.chart_path = ""

# -----------------------
# 🔹 Utility Functions
# -----------------------
def load_data():
    """
    Load your dataset here.
    Replace this with your actual data-loading logic.
    """
    # Example: returning a simple random dataframe
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "Fraud": [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

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

        # You may adjust the model name and prompt content as needed
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

# -----------------------
# 🔹 Callback Classes
# -----------------------
class StreamlitCallback(BaseCallback):
    """
    Callback to display AI-generated code in the left column.
    """
    def __init__(self, left_code_container, left_response_container) -> None:
        self.left_code_container = left_code_container
        self.left_response_container = left_response_container
        self.generated_code = ""

    def on_code(self, response: str):
        """Displays AI-generated code"""
        self.generated_code = response
        self.left_code_container.code(response, language="python")

    def get_generated_code(self):
        """Returns the last generated code"""
        return self.generated_code

class StreamlitResponse(ResponseParser):
    """
    Custom response parser to handle DataFrames, Plots, and Other outputs.
    Expects 'result["value"]' to be either a file path, a Matplotlib figure, or text.
    """
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        global left_response_container
        left_response_container.dataframe(result["value"])
        st.session_state.chart_generated = False

    def format_plot(self, result):
        global left_response_container
        chart_path = os.path.abspath("temp_chart.png")

        # If result["value"] is a string (filepath)
        if isinstance(result["value"], str):
            existing_chart_path = os.path.abspath(result["value"])
            if existing_chart_path == chart_path:
                # Already temp_chart.png
                st.session_state.chart_generated = True
                st.session_state.chart_path = chart_path
                left_response_container.image(chart_path)
            else:
                # It's a different file path
                try:
                    shutil.copy(existing_chart_path, chart_path)
                    st.session_state.chart_generated = True
                    st.session_state.chart_path = chart_path
                    left_response_container.image(chart_path)
                except Exception as e:
                    st.error(f"⚠️ Error copying chart: {e}")
                    st.session_state.chart_generated = False

        # If it's a Matplotlib figure
        elif isinstance(result["value"], plt.Figure):
            result["value"].savefig(chart_path)
            st.session_state.chart_generated = True
            st.session_state.chart_path = chart_path
            left_response_container.image(chart_path)

        else:
            st.error("⚠️ Unexpected chart format returned.")
            st.session_state.chart_generated = False

    def format_other(self, result):
        global left_response_container
        left_response_container.markdown(f"### 📌 AI Insight\n\n{result['value']}")
        st.session_state.chart_generated = False

# -----------------------
# 🔹 Streamlit App Layout
# -----------------------
st.write("# Chat with Credit Card Fraud Dataset 🦙")

# Load data
df = load_data()

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

query = st.text_area("🗣️ Ask a Data Analysis Question (Left Side AI):")

# Create two columns for layout
left_column, right_column = st.columns([1, 1])

# These containers will be used in the StreamlitCallback & StreamlitResponse
left_code_container = left_column.empty()
left_response_container = left_column.empty()

# 🎯 1) Process the user query with PandasAI (Left Side)
if query:
    llm = PandasOpenAI(api_token=os.environ["OPENAI_API_KEY"])
    callback_handler = StreamlitCallback(left_code_container, left_response_container)

    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,  # Pass the class, not instance
            "callback": callback_handler,
        },
    )

    answer = query_engine.chat(query)

    generated_code = callback_handler.get_generated_code()
    st.session_state.generated_code = generated_code
    st.session_state.editor_code = generated_code

# 🎯 2) GPT-4 Vision Interpreter Button (Left Side)
if st.session_state.chart_generated:
    # Only show the button if a chart was created
    gpt4_vision_button = left_column.button("🧐 GPT-4 Vision Interpreter (Left Side)")

    if gpt4_vision_button:
        # Analyze the chart using GPT-4 Vision
        ai_insight = analyze_chart_with_openai(st.session_state.chart_path)
        left_response_container.markdown(f"### 📊 AI Chart Breakdown\n\n{ai_insight}")
else:
    left_column.button("🧐 GPT-4 Vision Interpreter (Left Side)", disabled=True)

# 🎯 3) Right Side: Editor + "Run Code" Button
if st.session_state.generated_code:
    with right_column:
        # Editor
        edited_code = st_ace(
            value=st.session_state.editor_code,
            language="python",
            theme="tomorrow",
            key="code_editor",
            height=750,
        )
        st.session_state.editor_code = edited_code

        # Run Code Button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")  # placeholder
        with col2:
            run_button = st.button("🚀 Run Code")

        # Chart / Result Container Below Editor
        right_chart_container = st.empty()

        if run_button:
            try:
                exec_locals = {}

                # Execute the edited code, providing common data modules
                exec(
                    st.session_state.editor_code,
                    {"df": df, "pd": pd, "plt": plt, "st": st, "sns": sns},
                    exec_locals
                )

                # If the user code defines "analyze_data()", we can call it
                if "analyze_data" in exec_locals:
                    analyze_func = exec_locals["analyze_data"]
                    params = inspect.signature(analyze_func).parameters

                    if "user_input" in params:
                        result = analyze_func([df], user_input="")
                    else:
                        result = analyze_func([df])

                    # Display the result in the right_chart_container
                    if result["type"] == "plot":
                        right_chart_container.image(result["value"])
                    elif result["type"] == "dataframe":
                        right_chart_container.dataframe(result["value"])
                    elif result["type"] == "string":
                        right_chart_container.markdown(f"### 📌 AI Insight\n\n{result['value']}")
                    elif result["type"] == "number":
                        right_chart_container.write(f"Result: {result['value']}")

            except Exception as e:
                st.error(f"⚠️ Error running the code: {e}")
