import os
import time
import base64
import shutil
import io
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from data import load_data
# ------------------------------
# PandasAI + Callbacks
# ------------------------------
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.callbacks import BaseCallback
from pandasai.responses.response_parser import ResponseParser

# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}
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
.stTextArea, .stDataFrame, .st-code-block {
    border: 1px solid #dadada;
    border-radius: 4px;
    background-color: #fafafa;
    padding: 0.5em;
}
.css-1cpxqw2, .css-1q8dd3e, .stButton button {
    background-color: #006aff !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight:600 !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# OpenAI Setup
# ------------------------------
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# or openai.api_key = st.secrets["OPENAI_API_KEY"]
# if using standard openai lib calls

# Example Assistant ID & Thread (if you’re using a custom RAG endpoint).
ASSISTANT_ID = "asst_XXXX"
THREAD_ID = "thread_YYYY"

# -------------
# Session State
# -------------
if "chart_path" not in st.session_state:
    st.session_state.chart_path = None
if "pandas_code" not in st.session_state:
    st.session_state.pandas_code = None
if "assistant_interpretation" not in st.session_state:
    st.session_state.assistant_interpretation = None
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "vision_result" not in st.session_state:
    st.session_state.vision_result = None

# ------------------------------
# 1) Load Some Example Data
# ------------------------------
# @st.cache_data
# def load_data():
#     # For demonstration, let's create a small random dataset
#     import numpy as np
#     df_example = pd.DataFrame({
#         "transaction_amount": np.random.exponential(scale=100, size=1000),
#         "fraud_label": np.random.choice([0,1], size=1000, p=[0.95, 0.05]),
#         "customer_age": np.random.randint(18, 80, size=1000),
#     })
#     return df_example

df = load_data()

# ------------------------------
# 2) Utility: Encode Image
# ------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
        print("Base64 Image Length: ", len(encoded_str))
        return encoded_str

# ------------------------------
# 3) GPT-4 Vision Analysis
# ------------------------------
def analyze_chart_with_openai(image_path, user_request, assistant_summary, code):
    if not os.path.exists(image_path):
        return "⚠️ No chart found."

    base64_image = encode_image(image_path)

    uploaded_image_url = f'file://{image_path}'

    combined_prompt = f"""
You are an expert data analyst with additional chart context.
Here is the user’s original request:
"{user_request}"

Assistant summarized the request as:
"{assistant_summary}"

PandasAI generated this chart code:
(This code shows how the data was plotted, including x/y columns, grouping, etc.)

Now you have the final chart attached as an image.
Please provide a thorough interpretation of this chart:
1) Describe the chart type and axes.
2) Identify any trends, peaks, or patterns.
3) Tie it back to the user’s request about fraud detection or other relevant context.
4) Provide next steps or insights.

Avoid making assumptions beyond what the data or chart shows.
"""

    try:
        # Use chat.completions (because "gpt-3.5-turbo" is a chat model)
        response = openai.chat.completions.create(
            model="gpt-4.5-preview",   # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": """"✅ 
    **Explicitly defines what 0 and 1 mean** → No assumptions.  
✅ **Forces a per-category breakdown** → Prevents misgrouping.  
✅ **Uses generated code and dataset query as extra context** → Aligns with PandasAI’s output.  
✅ **Includes clear analysis steps** → Guides GPT-4 to focus on key aspects."}
"""},
            {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{combined_prompt}\n\nNow, analyze the attached image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=750,
            temperature=0.3,
        )
        # Extract the chat response
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT-4 Vision endpoint: {e}"
# ------------------------------
# 4) RAG Assistant Interpretation
# ------------------------------
def get_assistant_interpretation(user_input):
    """
    Summarizes the user's request, focusing on what the user wants to see or analyze.
    """
    prompt = f"""
Summarize the user's request from the following text without altering its core meaning.
Focus on what the user wants to see or analyze in the data:
---
{user_input}
---
"""

    try:
        # Again, use the Chat endpoint for a chat model (like gpt-3.5-turbo)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        st.warning(f"Error in get_assistant_interpretation: {e}")
        return "Could not interpret user request."

# ------------------------------
# 5) Custom PandasAI Callbacks
# ------------------------------
class StreamlitCallback(BaseCallback):
    def __init__(self) -> None:
        self.generated_code = ""

    def on_code(self, response: str):
        self.generated_code = response

    def get_generated_code(self):
        return self.generated_code

class StreamlitResponse(ResponseParser):
    # Remove the __init__ method entirely, let the parent handle default context.

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        st.session_state.chart_path = None

    def format_plot(self, result):
        chart_path = os.path.abspath("temp_chart.png")
        if isinstance(result["value"], str):
            existing_chart_path = os.path.abspath(result["value"])
            if existing_chart_path == chart_path:
                st.session_state.chart_path = chart_path
            else:
                try:
                    shutil.copy(existing_chart_path, chart_path)
                    st.session_state.chart_path = chart_path
                except Exception as e:
                    st.error(f"Error copying chart: {e}")
                    st.session_state.chart_path = None
        elif isinstance(result["value"], plt.Figure):
            result["value"].savefig(chart_path)
            st.session_state.chart_path = chart_path
        else:
            st.error("Unexpected chart format returned.")
            st.session_state.chart_path = None

    def format_other(self, result):
        st.markdown(f"### AI Insight:\n\n{result['value']}")
        st.session_state.chart_path = None

# ------------------------------
# MAIN PAGE: Integrated Example
# ------------------------------

if __name__ == "__main__":

    st.title("Advanced PandasAI + Vision GPT-4 Workflow")
    st.write("Enter a question to generate a chart, then interpret it with extra context.")

    user_query = st.text_input("Enter your question (e.g., 'Plot a scatter chart of transaction_amount vs. customer_age, colored by fraud_label')")

    if user_query:
        st.session_state.user_query = user_query  # store original user request

        # 1) Have the assistant interpret the query
        with st.spinner("Assistant interpreting your request..."):
            interpretation = get_assistant_interpretation(user_query)
            st.session_state.assistant_interpretation = interpretation

        st.subheader("Assistant Interpretation")
        st.write(interpretation)

        # 2) Build prompt for PandasAI
        # We can combine user_query + interpretation,
        # but you said you want to keep the user’s request mostly the same.
        # We'll do something simple like:

        combined_prompt = f"""
    User wants the following analysis (summarized):
    {interpretation}
    
    Now please create a plot or data analysis responding to the user request:
    {user_query}
    """

        # 3) Call PandasAI
        st.subheader("PandasAI Generating Chart Code")

        code_callback = StreamlitCallback()

        #response_parser = StreamlitResponse()

        code_callback = StreamlitCallback()

        llm = PandasOpenAI(api_token=st.secrets["OPENAI_API_KEY"])  # or a different LLM if desired

        sdf = SmartDataframe(
            df,
            config={
                "llm": llm,
                "callback": code_callback,
                "response_parser": StreamlitResponse,
            },
        )

        with st.spinner("Generating chart..."):
            answer = sdf.chat(combined_prompt)


        # 4) Grab the generated code
        st.session_state.pandas_code = code_callback.get_generated_code()

        st.subheader("AI Response / Explanation")
        st.write(answer)

        # 7) Retrieve & show the generated code
        code = code_callback.get_generated_code()

        if code:
            st.markdown("### Generated Code:")
            st.code(code, language="python")

        # 5) If a chart got created, show it
        if st.session_state.chart_path and os.path.exists(st.session_state.chart_path):
            st.image(st.session_state.chart_path, caption="Chart from PandasAI")
        else:
            st.info("No chart was generated or it couldn't be saved.")

    # ----------------------------------
    # GPT-4 Vision Interpretation Button
    # ----------------------------------
    if st.session_state.chart_path and os.path.exists(st.session_state.chart_path):
        st.markdown("---")
        st.write("### Final Step: GPT-4 Vision Interpretation")
        if st.button("Analyze This Chart with Extra Context"):
            with st.spinner("Analyzing chart..."):
                result = analyze_chart_with_openai(
                    image_path=st.session_state.chart_path,
                    user_request=st.session_state.user_query,
                    assistant_summary=st.session_state.assistant_interpretation,
                    code=st.session_state.pandas_code,
                )
                st.session_state.vision_result = result

        if st.session_state.vision_result:
            st.write("**GPT-4 Vision Analysis**:")
            st.markdown(st.session_state.vision_result)
    else:
        st.write("*(No chart to interpret yet.)*")
