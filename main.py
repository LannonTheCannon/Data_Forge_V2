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
from streamlit_ace import st_ace
import inspect
from streamlit_elements import elements, dashboard, mui, html
# from data import load_data
from resources.documentation_page_1 import documentation_page
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
# from streamlit_flow import streamlit_flow
# from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
# from streamlit_flow.state import StreamlitFlowState
# from streamlit_flow.layouts import RadialLayout


# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
# ------------------------------
# OpenAI Setup
# ------------------------------
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# or openai.api_key = st.secrets["OPENAI_API_KEY"]
# if using standard openai lib calls

# Example Assistant ID & Thread (if you’re using a custom RAG endpoint).
ASSISTANT_ID = "asst_XXXX"
THREAD_ID = "thread_YYYY"

layout = [
    # (element_identifier, x, y, w, h, additional_props...)
    dashboard.Item("first_item", 0, 0, 2, 2),  # Draggable & resizable by default
    dashboard.Item("second_item", 2, 0, 2, 2),  # Not draggable
    dashboard.Item("third_item", 0, 2, 1, 1),    # Not resizable
    dashboard.Item("chart_item", 4, 0, 3, 3)   # Our new chart card
]

# ----------------- Color Palette ------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

if "chart_path" not in st.session_state:
    st.session_state.chart_path = None
if 'editor_code' not in st.session_state:
    st.session_state.editor_code = ''
if "pandas_code" not in st.session_state:
    st.session_state.pandas_code = None
if "user_code_result" not in st.session_state:
    st.session_state.user_code_result = None
if "assistant_interpretation" not in st.session_state:
    st.session_state["assistant_interpretation"] = None
if "user_query" not in st.session_state:
    st.session_state['user_query'] = ""
if "vision_result" not in st.session_state:
    st.session_state.vision_result = None
if 'df' not in st.session_state:
    st.session_state.df = None
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None
if "question_list" not in st.session_state:
    st.session_state['question_list'] = []
if "metadata_string" not in st.session_state:
    st.session_state['metadata_string'] = ""
if "trigger_assistant" not in st.session_state:
    st.session_state["trigger_assistant"] = False  # Ensures assistant runs when needed
if "saved_charts" not in st.session_state:
    st.session_state['saved_charts'] = []

def reset_session_variables():
    # reset session state variables

    st.session_state["assistant_interpretation"] = None
    st.session_state["pandas_code"] = None
    st.session_state["chart_path"] = None
    st.session_state["vision_result"] = None
    st.session_state["user_query"] = ""  # Clear the input field as well
    st.session_state["trigger_assistant"] = False

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

def to_base64(path_to_png):
    with open(path_to_png, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
        print("Base64 Image Length: ", len(encoded_str))
        return encoded_str

def analyze_chart_with_openai(image_path, user_request, assistant_summary, code, meta):
    if not os.path.exists(image_path):
        return "⚠️ No chart found."

    base64_image = encode_image(image_path)
    uploaded_image_url = f'file://{image_path}'

#4) Provide next steps or insights.

    combined_prompt = f"""    
You are an expert data analyst with additional chart context and metadata based on the actual dataset provided
"{meta}"

Here is the user’s original request:
"{user_request}"

Assistant summarized the request as:
"{assistant_summary}"

PandasAI generated this chart code:
(This code shows how the data was plotted, including x/y columns, grouping, etc.)
{code}

Now you have the final chart attached as an image.
Please provide a thorough interpretation of this chart:
1) Describe the chart type and axes.
2) Identify any trends, peaks, or patterns.
3) Tie it back to the user’s request.

Avoid making assumptions beyond what the data or chart shows.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",   # or "gpt-4" if you have access
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

def generate_multiple_question_sets():
    question_set_1 = get_list_questions()
    question_set_2 = get_list_questions()
    question_set_3 = get_list_questions()

    return question_set_1, question_set_2, question_set_3

def identify_common_questions(question_set_1, question_set_2, question_set_3):
    """Uses AI to analyze the three sets and identify the most common questions."""
    messages = [
        {"role": "system", "content": f"""You are an expert data analyst assistant. Your task is to identify the most relevant and commonly occurring questions from three generated question sets.

- Find the **most frequent** or **semantically similar** questions across the three sets.
- Ensure a **balanced mix** of insights (time trends, correlations, distributions, categorical comparisons).
- Avoid redundancy while keeping the most valuable questions.

Here are the three sets of generated questions:
Set 1:
{question_set_1}

Set 2:
{question_set_2}

Set 3:
{question_set_3}

Identify the **top 5 most relevant** questions based on frequency and analytical value."""},
        {"role": "user",
         "content": f"Here are the question sets:\n\nSet 1: {question_set_1}\nSet 2: {question_set_2}\nSet 3: {question_set_3}\n\nPlease provide the 5 most common and relevant questions."}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800
        )
        common_questions = response.choices[0].message.content.strip().split("\n")
        return [q.strip() for q in common_questions if q.strip()]

    except Exception as e:
        return [f"Error identifying common questions: {str(e)}"]

def get_list_questions():
    """ Generate a list of questions based on the uploaded dataset """
    # Ensure we have metadata before proceeding
    if "df_summary" not in st.session_state or st.session_state.df_summary is None:
        return ["No dataset summary found. Please upload a dataset first."]

    metadata = {
        "columns": list(st.session_state.df_summary.columns),  # Extract column names
        "summary": st.session_state.df_summary.to_dict(),  # Convert summary stats to a dictionary
        "row_count": st.session_state.df.shape[0] if st.session_state.df is not None else 0
    }

    metadata_string = f"Columns: {', '.join(metadata['columns'])}\nTotal Rows: {metadata['row_count']}\nSummary Stats: {metadata['summary']}"

    # Define chat messages format
    messages = [
        {"role": "system", "content": """You are a data analyst assistant that generates insightful and structured visualization questions based on dataset metadata. 

- These questions must be optimized for use with the Pandas AI Smart DataFrame.
- They should be **concise** and **direct**, avoiding overly descriptive or wordy phrasing.
- Each question should focus on **specific relationships** or **trends** that can be effectively visualized.
- Prioritize **correlation, distribution, time-based trends, and categorical comparisons** in the dataset.
- Format them as a numbered list.

Given the following dataset metadata:
{metadata_string}

Generate 5 structured questions that align with best practices in data analysis and visualization."""},
        {"role": "user", "content": f"""Given this dataset metadata: {metadata_string}, generate 5 insightful data analysis questions."""}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800
        )

        # return response.choices[0].message.content
        questions = response.choices[0].message.content.strip().split("\n")

        return (metadata_string, [q.strip() for q in questions if q.strip()])

    except Exception as e:
        return [f"Error generating questions: {str(e)}"]

def get_assistant_interpretation(user_input, metadata):
    prompt = f"""
*Reinterpret the user’s request into a clear, visualization-ready question that aligns with the dataset’s 
structure and is optimized for charting. Focus on extracting the core analytical intent, ensuring the output 
is compatible with PandasAI’s ability to generate meaningful graphs.

Here is the user's original query: {user_input}
Here is the dataset metadata "{metadata}"

Abstract away ambiguity—Do not take the request too literally. Instead, refine it to emphasize patterns, trends, 
distributions, or comparisons that can be effectively represented visually.
Ensure clarity for PandasAI—Frame the question in a way that translates naturally into a visualization rather 
than a direct data lookup or overly complex query.
Align with the dataset’s metadata—Use insights from the metadata to guide the interpretation, ensuring that the 
suggested visualization is relevant to the data type (e.g., time series trends, categorical distributions, correlations).
Prioritize chart compatibility—Reframe vague or broad queries into specific, actionable visual analysis 
that can be represented using line charts, bar graphs, scatter plots, or heatmaps.*
"""

    try:
        # Again, use the Chat endpoint for a chat model (like gpt-3.5-turbo)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
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

def handle_layout_change(updated_layout):
    st.session_state['dashboard_layout'] = updated_layout
    print("Updated layout in the app:", updated_layout)  # Goes to the app UI

def show_dashboard():
    # If dashboard layout exists in session state, use it;
    # otherwise, generate a layout dynamically based ont he number of saved charts.
    saved_charts = st.session_state.get('saved_charts', [])

    # Initialize dashboard_layout only once.

    if 'dashboard_layout' not in st.session_state:
        st.session_state.dashboard_layout = []

    # If there are more saved charts than dashboard items,
    # append new default items for the additional charts.

    if len(st.session_state.dashboard_layout) < len(saved_charts):
        start_idx = len(st.session_state.dashboard_layout)
        for idx in range(start_idx, len(saved_charts)):
            # Set default position and size (uniform for new charts).
            x = (idx % 3) * 3  # Three cards per row.
            y = (idx // 3) * 3
            st.session_state.dashboard_layout.append(
                dashboard.Item(f'chart_item_{idx}', x, y, 3, 3,
                               isDraggable=True, isResizable=True),
            )

    # Use the dashboard_layout from session state.
    dashboard_layout = st.session_state.dashboard_layout
    print("Dashboard Layout:", dashboard_layout)  # Debug: see current layout
    # Build the dashboard grid
    with elements("dashboard"):
        with dashboard.Grid(dashboard_layout, onLayoutChange=handle_layout_change, key='my_dashboard_grid'):
            # For each saved chart, create a draggable/resizable Paper element.
            for idx, chart_path in enumerate(saved_charts):
                with mui.Paper(key=f'chart_item_{idx}',
                               sx={
                                   'minWidth': '200px',
                                   'minHeight': '200px',
                                   'overflow': 'hidden',
                                   'display': 'flex',
                                   'justifyContent': 'center',
                                   'alignItems': 'center'
                               }):

                    if os.path.exists(chart_path):
                        b64 = to_base64(chart_path)
                        html.Img(
                            src=f"data:image/png;base64,{b64}",
                            style={
                                "width": "100%",
                                "height": "100%",
                                "objectFit": "cover"  # This scales the image to fill the card nicely.
                            }
                        )
                    else:
                        mui.Typography('Chart file not found')

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

PAGE_OPTIONS = [
    'Data Upload',
    'Mind Mapping',
    'Pandas Viz',
    "Code Editor",
    'Dashboard',
    'Documentation'
]

page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('Upload your own Dataset!')
        uploaded_file = st.file_uploader('Upload CSV and Excel Here', type=['csv', 'excel'])

        if uploaded_file is not None:
            # Load data into session state
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_preview = df.head()
                st.session_state.df_summary = df.describe()

        # Display preview & summary if data exists
        if st.session_state.df is not None:
            st.write("### Data Preview")
            st.write(st.session_state.df_preview)

            st.write("### Data Summary")
            st.write(st.session_state.df_summary)

    elif page == 'Mind Mapping':
        st.title('Mind Mapping + OpenAI Ensemble Completions')
        st.write('Identify a category of the dataset you would like to explore.')

        # New Feature: Suggested Categories
        st.write('### Suggested Questions')

        if "question_list" not in st.session_state:
            st.session_state["question_list"] = []

        if st.button("Generate Most Relevant Questions"):
            st.write("Generating multiple question sets...")

            question_set_1, question_set_2, question_set_3 = generate_multiple_question_sets()
            common_questions = identify_common_questions(question_set_1, question_set_2, question_set_3)

            st.session_state["question_list"] = common_questions
            print(st.session_state['question_list'])

        st.write("### Most Relevant Questions:")
        for idx, question in enumerate(st.session_state["question_list"]):
            # Okay at this point we need to do some list/ string manipulation
            if (question[0]) in ['1','2','3','4','5']:
                if st.button(f" 🔍 {question}"):
                    reset_session_variables()
                    st.session_state["user_query"] = question
                    st.session_state["trigger_assistant"] = True  # Ensure assistant runs
            else:
                st.write(f"{question}")

    elif page == 'Pandas Viz':

        st.title('Pandas Visualization')

        new_user_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("user_query", "")
        )

        if new_user_query != st.session_state['user_query']:  # if a new query is entered
            reset_session_variables()
            st.session_state['user_query'] = new_user_query

        if st.button('generate interpretation'):
            st.session_state["trigger_assistant"] = True

        # if st.session_state['trigger_assistant']:

        if st.session_state['trigger_assistant']:
            with st.spinner("Assistant interpreting your request..."):
                interpretation = get_assistant_interpretation(new_user_query, st.session_state['metadata_string'])
                st.session_state["assistant_interpretation"] = interpretation  # Store in session state
                st.session_state["trigger_assistant"] = False
        else:
            interpretation = st.session_state["assistant_interpretation"]  # Use existing interpretation

        if st.session_state.get('assistant_interpretation'):
            st.subheader("Assistant Interpretation")
            st.write(interpretation)

            combined_prompt = f"""
        User wants the following analysis (summarized):
        {interpretation}
        
        Now please create a plot or data analysis responding to the user request:
        {new_user_query}
        """

            # 3) Call PandasAI
            st.subheader("PandasAI Generating Chart Code")
            code_container = st.container()
            response_container = st.container()
            code_callback = StreamlitCallback(code_container, response_container)

            #response_parser = StreamlitResponse()

            llm = PandasOpenAI(api_token=st.secrets["OPENAI_API_KEY"])  # or a different LLM if desired
            # client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            # llm = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

            sdf = SmartDataframe(
                st.session_state.df,
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
            st.session_state.editor_code = st.session_state.pandas_code

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
                        meta=st.session_state.metadata_string,
                    )
                    st.session_state.vision_result = result

            if st.session_state.vision_result:
                st.write("**GPT-4 Vision Analysis**:")
                st.markdown(st.session_state.vision_result)
        else:
            st.write("*(No chart to interpret yet.)*")

    elif page == "Code Editor":
        st.markdown('<h1 class="big-title">User Code Editor & Execution</h1>', unsafe_allow_html=True)

        if st.session_state.pandas_code:
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
                    exec(st.session_state.editor_code, {"df": st.session_state.df, "pd": pd, "plt": plt, "st": st, "sns": sns},
                         exec_locals)

                    if "analyze_data" in exec_locals:
                        analyze_func = exec_locals["analyze_data"]
                        params = inspect.signature(analyze_func).parameters

                        if "user_input" in params:
                            result = analyze_func([st.session_state.df], user_input="")
                        else:
                            result = analyze_func([st.session_state.df])

                        st.session_state.user_code_result = result

                except Exception as e:
                    st.error(f"⚠️ Error running the code: {e}")

            if st.session_state.user_code_result:
                result = st.session_state.user_code_result
                if result["type"] == "plot":
                    st.markdown('<p class="section-header">User-Generated Plot</p>', unsafe_allow_html=True)
                    st.image(result["value"])
                    # ↓↓↓ ADD THIS “SAVE CHART” BUTTON ↓↓↓
                    if st.button("💾 Save Chart to Dashboard"):
                        if st.session_state.chart_path and os.path.exists(st.session_state.chart_path):
                            import datetime
                            # generate a unique filename using the current timestamp
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            new_filename = f'chart_{timestamp}.png'
                            try:
                                # copy the current temp chart file to the new unique file
                                shutil.copy(st.session_state.chart_path, new_filename)
                                # append the new filename to the saved_charts list
                                st.session_state["saved_charts"].append(new_filename)
                                st.success("chart saved to dashboard")
                            except Exception as e:
                                st.error(f'Error saving chart: {e}')
                        else:
                            st.error('No temporary chart available to save.')

                elif result["type"] == "dataframe":
                    st.markdown('<p class="section-header">DataFrame Output</p>', unsafe_allow_html=True)
                    st.dataframe(result["value"])
                elif result["type"] == "string":
                    st.markdown(f"### 📌 AI Insight\n\n{result['value']}")
                elif result["type"] == "number":
                    st.write(f"Result: {result['value']}")
        else:
            st.info("No generated code yet. Please go to 'PandasAI Insights' and enter a query first.")

    elif page == 'Dashboard':
        st.title("Dashboard of Saved Charts")
        show_dashboard()

    elif page == 'Documentation':
        documentation_page()
