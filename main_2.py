import os
import time
import base64
import numpy as np
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
# Streamlit FLOW Layout
# ------------------------------
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout
import random
from uuid import uuid4
# ------------------------------------------------------------------

# Importing data science team stuff
# import streamlit.components.v1 as components
# from pathlib import Path
# import html
#
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain_openai import ChatOpenAI
#
# from ai_data_science_team.ds_agents import EDAToolsAgent
# from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
# from ai_data_science_team.utils.plotly import plotly_from_dict

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
    dashboard.Item("third_item", 0, 2, 1, 1),  # Not resizable
    dashboard.Item("chart_item", 4, 0, 3, 3)  # Our new chart card
]

# ----------------- Color Palette ------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# ------------------- Initialize Session -------------------
if "curr_state" not in st.session_state:
    # Prepare root node. We'll store the dataset metadata in "full_question" if we have it.
    dataset_label = st.session_state.get("dataset_name", "Dataset")

    # We'll call it "S0" for the section path.
    root_node = StreamlitFlowNode(
        "S0",
        (0, 0),
        {
            "section_path": "S0",
            "short_label": "ROOT",
            "full_question": "",  # We'll fill in once we have metadata
            "content": dataset_label
        },
        "input",
        "right",
        style={"backgroundColor": COLOR_PALETTE[0]}
    )
    st.session_state.curr_state = StreamlitFlowState(nodes=[root_node], edges=[])
    st.session_state.expanded_nodes = set()

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
if 'dataset_name' not in st.session_state:
    st.session_state['dataset_name'] = ""
# For storing each node the user has clicked (to display full question)
if "clicked_questions" not in st.session_state:
    st.session_state.clicked_questions = []
# For controlling repeated expansions
if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()

# ------------------- Color Setup -------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]


# ------------------ STEP 1 DATA UPLOAD -------------------#
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

# ------------------- Mind Mapping Logic -------------------

def generate_root_summary_question(metadata_string: str) -> str:
    """
    Uses OpenAI to produce a single-sentence question or statement that describes
    or summarizes the dataset metadata.
    :param metadata_str:
    :return: beautified metadata str
    """
    if not metadata_string:
        return "Overview of the dataset"

    messages = [
        {
            "role": "system",
            "content": """You are a data summarizer. 
               Given dataset metadata, produce a single-sentence question 
               or statement that captures the main theme or focus of the dataset. 
               Keep it short (one sentence) and neutral."""
        },
        {
            "role": "user",
            "content": f"Dataset metadata:\n{metadata_string}\n\nPlease provide one short question about the dataset."
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50  # Enough for a short response
        )
        text = response.choices[0].message.content.strip()
        # Just in case the model returns multiple lines, combine them or take first line
        lines = text.split("\n")
        return lines[0].strip()
    except Exception as e:
        # Fallback if there's an error
        return "What is the primary focus of this dataset?"


def get_list_questions(context: str):
    """
    Generate 5 questions given the dataset metadata plus
    the parent's context (if any).
    """
    if "df_summary" not in st.session_state or st.session_state.df_summary is None:
        return ["No dataset summary found. Please upload a dataset first."]

    # Build base metadata from session
    metadata = {
        "columns": list(st.session_state.df_summary.columns),
        "summary": st.session_state.df_summary.to_dict(),
        "row_count": st.session_state.df.shape[0] if st.session_state.df is not None else 0
    }
    metadata_string = (
        f"Columns: {', '.join(metadata['columns'])}\n"
        f"Total Rows: {metadata['row_count']}\n"
        f"Summary Stats: {metadata['summary']}"
    )

    # You can combine the parent's 'context' (question text) with the metadata
    # so the AI knows what the user is focusing on.
    combined_context = (
        f"Parent's context/question: {context}\n\n"
        f"Dataset metadata:\n{metadata_string}"
    )

    messages = [
        {
            "role": "system",
            "content": """You are a data analyst assistant that generates concise, structured, 
            and insightful visualization questions. Each question should focus on 
            specific data relationships or trends, referencing the relevant columns when possible.
            You are a data analyst assistant that generates insightful and structured visualization questions based on dataset metadata. 

            - These questions must be optimized for use with the Pandas AI Smart DataFrame.
            - They should be **concise** and **direct**, avoiding overly descriptive or wordy phrasing.
            - Each question should focus on **specific relationships** or **trends** that can be effectively visualized.
            - Prioritize **correlation, distribution, time-based trends, and categorical comparisons** in the dataset.
            - Format them as a numbered list.

            Given the following dataset metadata:
            {metadata_string}

            Generate 4 structured questions that align with best practices in data analysis and visualization. 

            """
        },
        {
            "role": "user",
            "content": f"Given the context and metadata below, generate 4 short data analysis questions:\n{combined_context}"
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800
        )
        raw = response.choices[0].message.content.strip()
        questions = raw.split("\n")
        # Filter out empty lines
        cleaned = [q.strip() for q in questions if q.strip()]
        return cleaned

    except Exception as e:
        return [f"Error generating questions: {str(e)}"]


def generate_multiple_question_sets(parent_context: str):
    """
    Calls get_list_questions 3 times to create 3 sets of questions
    using the parent's context (full question or dataset metadata).
    """
    q1 = get_list_questions(parent_context)
    q2 = get_list_questions(parent_context)
    q3 = get_list_questions(parent_context)
    return q1, q2, q3


def identify_common_questions(question_set_1, question_set_2, question_set_3):
    """
    Uses AI to find the most relevant, commonly occurring questions
    across the three sets. We'll return the top 4.
    """
    joined_1 = "\n".join(question_set_1)
    joined_2 = "\n".join(question_set_2)
    joined_3 = "\n".join(question_set_3)

    messages = [
        {
            "role": "system",
            "content": """You are an expert data analyst assistant. 
               Identify the 4 most relevant and commonly occurring questions 
               from the three sets. Provide exactly 4 lines with no numbers,
               bullet points, or additional text. """
        },
        {
            "role": "user",
            "content": f"""Set 1:\n{joined_1}\n
                            Set 2:\n{joined_2}\n
                            Set 3:\n{joined_3}\n
                            Please provide exactly 4 questions, each on its own line, with no numbering."""
        }
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800
        )
        raw = response.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        # If more than 4 lines come back, just take the first 4
        return lines[:4]
    except Exception as e:
        return [f"Error identifying common questions: {str(e)}"]


def paraphrase_questions(questions):
    """
    Paraphrase multiple questions into short labels or titles.
    Returns a list of short strings (one for each question).
    """
    joined = "\n".join(questions)
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that transforms full-length data analysis
            questions into short, descriptive labels (5-8 words max). 
            Example: 'How do sales vary by region?' -> 'Sales by Region'."""
        },
        {
            "role": "user",
            "content": f"Please paraphrase each question into a short label:\n{joined}"
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        # Attempt to match them one-to-one
        # Ensure we only return as many paraphrases as we had questions
        paraphrased = lines[:len(questions)]
        return paraphrased
    except Exception as e:
        # On error, just return the original questions
        return questions


def get_section_path_children(parent_path: str, num_children=4):
    """
    Given a parent's section path like 'S0.1', produce a list:
    ['S0.1.1', 'S0.1.2', 'S0.1.3', 'S0.1.4'].
    """
    children = []
    for i in range(1, num_children + 1):
        new_path = f"{parent_path}.{i}"
        children.append(new_path)
    return children


def get_color_for_depth(section_path: str):
    """
    Depth = number of dots in the section path
    For example, S0 -> depth 0, S0.1 -> depth 1, S0.1.2 -> depth 2, etc.
    Then use your existing COLOR_PALETTE.
    """
    depth = section_path.count(".")
    return COLOR_PALETTE[depth % len(COLOR_PALETTE)]


def expand_node_with_questions(clicked_node):
    """
    1) Figure out the parent's context. If it's the root, context = dataset metadata.
       Otherwise, context = parent's full question.
    2) Generate 4 new questions via the ensemble approach.
    3) Paraphrase them for short labels.
    4) Create child nodes labeled with the parent's section_path + .1, .2, .3, .4.
    5) Connect them in the flow.
    6) Mark the parent node as expanded.
    7) Log the parent's full question in st.session_state.clicked_questions.
    """
    # 1) Get parent's full_question if it exists, else fallback to dataset metadata
    parent_full_question = clicked_node.data.get("full_question", "")
    if not parent_full_question:
        # Probably the root node
        parent_full_question = f"(Root) Dataset Metadata:\n{st.session_state.metadata_string}"

    # 2) Generate 3 sets of questions & pick the top 4
    q1, q2, q3 = generate_multiple_question_sets(parent_full_question)
    top_questions = identify_common_questions(q1, q2, q3)

    # 3) Paraphrase them for short node labels
    short_labels = paraphrase_questions(top_questions)

    # 4) Create the child nodes. Derive their section paths
    parent_path = clicked_node.data.get("section_path", "S0")  # Root is "S0" by default
    child_paths = get_section_path_children(parent_path, num_children=4)

    for i, child_path in enumerate(child_paths):
        # For each new question, create a node
        new_node_id = child_path  # We'll also use it for the node ID
        color = get_color_for_depth(child_path)

        child_full_question = top_questions[i] if i < len(top_questions) else "N/A"
        child_short_label = short_labels[i] if i < len(short_labels) else child_full_question[:30]

        new_node = StreamlitFlowNode(
            new_node_id,
            (random.randint(-100, 100), random.randint(-100, 100)),
            {
                "section_path": child_path,
                "short_label": child_short_label,
                "full_question": child_full_question,

                # This is the key part:
                # Set the node's displayed text to your short label.
                "content": f"**{child_short_label}**"
            },
            "default",
            "right",
            "left",
            style={"backgroundColor": color}
        )

        # Add to state
        st.session_state.curr_state.nodes.append(new_node)
        edge_id = f"{clicked_node.id}-{new_node_id}"
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(edge_id, clicked_node.id, new_node_id, animated=True)
        )

    # 5) Mark as expanded
    st.session_state.expanded_nodes.add(clicked_node.id)

    # 6) Log the parent's question if not already
    #    (So we can see which node was clicked in a table at the bottom.)
    #    Avoid duplicates if user repeatedly clicks the same node (shouldn't happen with 'expanded_nodes', but just in case).
    existing_paths = [q["section"] for q in st.session_state.clicked_questions]
    if parent_path not in existing_paths:
        st.session_state.clicked_questions.append({
            "section": parent_path,
            "short_label": clicked_node.data.get("short_label", clicked_node.data.get("content", parent_path)),
            "full_question": parent_full_question
        })


def expand_root_node(clicked_node):
    # Hard-boiled coded list of the top-level EDA themes (DATA ARCHETYPES)

    themes = [
        ('Distributions', "#FF6B6B"),
        ("Correlations", "#6BCB77"),
        ("Missingness", "#4D96FF"),
        ("Data Types", "#FFD93D")
    ]

    parent_path = clicked_node.data['section_path']

    for i, (theme_label, color) in enumerate(themes):
        child_id = f'{parent_path}.{i + 1}'
        new_node = StreamlitFlowNode(
            child_id,
            (random.randint(-100, 100), random.randint(-100, 100)),
            {
                "section_path": child_id,
                "short_label": theme_label,
                "full_question": "",  # We'll leave empty for now
                "content": f"**{theme_label}**",
                "node_type": "thematic"  # <-- Label these nodes as 'thematic'
            },
            "default",
            "right",
            "left",
            style={"backgroundColor": color}
        )

        st.session_state.curr_state.nodes.append(new_node)

        edge_id = f"{clicked_node.id}-{child_id}"
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(edge_id, clicked_node.id, child_id, animated=True)
        )

        # Mark parent as expanded
        st.session_state.expanded_nodes.add(clicked_node.id)

        # log the parent's question if not already
        parent_section = clicked_node.data["section_path"]
        existing_paths = [q["section"] for q in st.session_state.clicked_questions]
        if parent_section not in existing_paths:
            st.session_state.clicked_questions.append({
                "section": parent_section,
                "short_label": clicked_node.data.get("short_label", "ROOT"),
                "full_question": "Root node expanded"
            })


def get_node_depth(node_id):
    return node_id.count("_")  # each underscore = one level deeper

# ------------------- Mind Mapping Logic -------------------

def reset_session_variables():
    # reset session state variables

    st.session_state["assistant_interpretation"] = None
    st.session_state["pandas_code"] = None
    st.session_state["chart_path"] = None
    st.session_state["vision_result"] = None
    st.session_state["user_query"] = ""  # Clear the input field as well
    st.session_state["trigger_assistant"] = False


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

    # 4) Provide next steps or insights.

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
            model="gpt-4o-mini",  # or "gpt-4" if you have access
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
                {"role": "system",
                 "content": "You are a helpful data analysis assistant designed to extract the core intent of the user query and form a high value prompt that can be used 100% of the time for pandasai for charting code. "},
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
    'Tree Mapping V2',
    'Pandas Viz',
    "Code Editor",
    'Dashboard',
    'Documentation'
]

page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('Upload your own Dataset!')
        uploaded_file = st.file_uploader('Upload CSV or Excel here', type=['csv', 'excel'])

        if uploaded_file is not None:
            # Load data into session state
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_preview = df.head()
                st.session_state.df_summary = df.describe()

                # Save dataset name without extension
                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name
                st.write(dataset_name)

                # Rebuild a 'metadata_string' for the root node
                if st.session_state.df_summary is not None:
                    # Basic example of turning summary + columns into a string
                    cols = list(st.session_state.df_summary.columns)
                    row_count = st.session_state.df.shape[0]
                    st.session_state.metadata_string = (
                        f"Columns: {cols}\n"
                        f"Total Rows: {row_count}\n"
                        f"Summary Stats:\n{st.session_state.df_summary}"
                    )

                    # Produce a one-sentence question describing the dataset
                    root_question = generate_root_summary_question(st.session_state.metadata_string)

                    # Update the root node's data if it exists:
                    if st.session_state.curr_state.nodes:
                        root_node = st.session_state.curr_state.nodes[0]
                        root_node.data["full_question"] = root_question
                        # Optionally display it on the node itself:
                        root_node.data["content"] = "ROOT"  # or root_node.data["content"] = root_question
                        root_node.data["short_label"] = "ROOT"

        # Display preview & summary if data exists
        if st.session_state.df is not None:
            st.write("### Data Preview")
            st.write(st.session_state.df_preview)

            st.write("### Data Summary")
            st.write(st.session_state.df_summary)

    elif page == 'Tree Mapping V2':
        st.title('Tree Mapping + OpenAI Ensemble')

        # Sync root node label with updated dataset name if needed
        if st.session_state.get("dataset_name"):
            root_node = st.session_state.curr_state.nodes[0]
            if root_node.data["content"] != st.session_state["dataset_name"]:
                root_node.data["content"] = st.session_state["dataset_name"]

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Reset Mind Map"):
                # Rebuild the root node
                dataset_label = st.session_state.get("dataset_name", "Dataset")
                new_root = StreamlitFlowNode(
                    "S0",
                    (0, 0),
                    {
                        "section_path": "S0",
                        "short_label": "ROOT",
                        "full_question": st.session_state.metadata_string,
                        "content": dataset_label
                    },
                    "input",
                    "right",
                    style={"backgroundColor": COLOR_PALETTE[0]}
                )
                st.session_state.curr_state = StreamlitFlowState(nodes=[new_root], edges=[])
                st.session_state.expanded_nodes = set()
                st.session_state.clicked_questions = []
                st.rerun()

        # Render the flow
        st.session_state.curr_state = streamlit_flow(
            "mind_map",
            st.session_state.curr_state,
            layout=TreeLayout(direction="right"),
            fit_view=True,
            height=550,
            get_node_on_click=True,
            enable_node_menu=True,
            enable_edge_menu=True,
            show_minimap=False
        )

        # If a node was clicked, expand it (if not already expanded)
        clicked_node_id = st.session_state.curr_state.selected_id
        if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
            node_map = {n.id: n for n in st.session_state.curr_state.nodes}
            clicked_node = node_map.get(clicked_node_id)
            if clicked_node:
                node_type = clicked_node.data.get("node_type", "")

                if node_type == "root":
                    expand_root_node(clicked_node)
                else:
                    expand_node_with_questions(clicked_node)

            st.rerun()
        # Display a table of all clicked questions so far
        if st.session_state.clicked_questions:
            st.write("## Questions Clicked So Far")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            st.table(df_log)

    # elif page == 'Mind Mapping V1':
    #     st.title('Mind Mapping + OpenAI Ensemble Completions')
    #     st.write('Identify a category of the dataset you would like to explore.')
    #
    #     # New Feature: Suggested Categories
    #     st.write('### Suggested Questions')
    #
    #     if "question_list" not in st.session_state:
    #         st.session_state["question_list"] = []
    #
    #     if st.button("Generate Most Relevant Questions"):
    #         st.write("Generating multiple question sets...")
    #
    #         question_set_1, question_set_2, question_set_3 = generate_multiple_question_sets()
    #         common_questions = identify_common_questions(question_set_1, question_set_2, question_set_3)
    #
    #         st.session_state["question_list"] = common_questions
    #         print(st.session_state['question_list'])
    #
    #     st.write("### Most Relevant Questions:")
    #     for idx, question in enumerate(st.session_state["question_list"]):
    #         # Okay at this point we need to do some list/ string manipulation
    #         if (question[0]) in ['1','2','3','4','5']:
    #             if st.button(f" 🔍 {question}"):
    #                 reset_session_variables()
    #                 st.session_state["user_query"] = question
    #                 st.session_state["trigger_assistant"] = True  # Ensure assistant runs
    #         else:
    #             st.write(f"{question}")

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

            # response_parser = StreamlitResponse()

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
                    exec(st.session_state.editor_code,
                         {"df": st.session_state.df, "pd": pd, "plt": plt, "st": st, "sns": sns, "np": np},
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
