import numpy as np
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import json
from code_editor import code_editor
import plotly.graph_objects as go
import plotly.io as pio
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout
import random
# from standalone_projects.standalone_streamlit_flow_nodes.basic_streamlit_flow_nodes_4 import COLOR_PALETTE

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index", "clicked_questions", "dataset_name", "expanded_nodes", "mindmap_nodes"]:

    if key not in st.session_state:
        if key == "mindmap_nodes":
            st.session_state[key] = {}  # ✅ FIX: dictionary for object lookup
        elif key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "DATA_RAW"]:
            st.session_state[key] = None
        else:
            st.session_state[key] = []

class MindMapNode:
    def __init__(
        self,
        node_id: str,
        label: str,
        full_question: str,
        category: str,
        node_type: str,
        payload: str = "",
        parent_id: str = None,
        color: str = "#6BCB77"
    ):
        self.node_id = node_id
        self.label = label
        self.full_question = full_question
        self.category = category  # Distribution, Correlation, etc.
        self.node_type = node_type  # intent, exploratory, conclusive
        self.payload = payload
        self.parent_id = parent_id
        self.color = color
        self.expanded = False

    def mark_expanded(self):
        self.expanded = True

    def is_conclusive(self):
        return self.node_type == "conclusive"

    def can_expand(self):
        return not self.expanded and not self.is_conclusive()

    def to_streamlit_node(self):
        from streamlit_flow.elements import StreamlitFlowNode
        return StreamlitFlowNode(
            self.node_id,
            (random.randint(-100, 100), random.randint(-100, 100)),
            {
                "section_path": self.node_id,
                "short_label": self.label,
                "full_question": self.full_question,
                "category": self.category,
                "node_type": self.node_type,
                "payload": self.payload,
                "content": f"**{self.label}**"
            },
            "default",
            "right",
            "left",
            style={"backgroundColor": self.color}
        )

if "curr_state" not in st.session_state:
    # Prepare root node. We'll store the dataset metadata in "full_question" if we have it.
    dataset_label = st.session_state.get("dataset_name", "Dataset")

    root_id = "S0"
    root_mindmap_node = MindMapNode(
        node_id=root_id,
        label="ROOT",
        full_question="Overview of the dataset",
        category="Meta",
        node_type="root",
        color=COLOR_PALETTE[0]
    )

    # Register in memory
    st.session_state.mindmap_nodes[root_id] = root_mindmap_node

    # Add to Streamlit flow
    st.session_state.curr_state = StreamlitFlowState(
        nodes=[root_mindmap_node.to_streamlit_node()],
        edges=[]
    )
    st.session_state.expanded_nodes = set()




# ######################### Data Upload function ######################### #

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None


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
            model="gpt-4.1-mini",
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


# ######################### Mind Mappping function ######################### #

def get_list_questions(context: str):
    """
    Generate 4 questions given the dataset metadata plus
    the parent's context (if any).
    """
    if "df_summary" not in st.session_state or st.session_state.df_summary is None:
        return ["No dataset summary found. Please upload a dataset first."]

    combined_context = (
        f"Parent's context/question: {context}\n\n"
        f"Dataset metadata:\n{st.session_state.metadata_string}"
    )

    prompt = """
You are **Viz‑Detective‑GPT**.

Your task is to propose **exactly FOUR** analysis tasks that can each be turned into a
**basic visualisation**.  Follow these rules:

1. **Start with the insight → choose the chart.**  
   Think about the relationship the user might want to see first, then pick
   the simplest chart that reveals it.

2. Stick to these elementary chart families  
   • Histogram (distribution of one numeric column)  
   • Bar chart (count or aggregate of one categorical column)  
   • Grouped / Stacked bar (two categorical columns)  
   • Line chart (trend over an ordered or date column)  
   • Scatter plot (two numeric columns)  
   • Scatter + LOESS / best‑fit line  
   • Box plot (numeric‑by‑categorical)  
   • Heat‑map (correlation or contingency table)  
   • Violin / Strip plot 

   Only use ONE chart if the dataset’s columns make sense for it, create table of 
   INSIGHT if not. Make sure you are explicit in saying "generate a table" if you are look for a key 
   insight for example "What is the highest salary in the dataset" or something like that. 

3. **Column discipline**  
   Use **only** the column names provided in the metadata.  
   Never invent new columns; never rename existing ones.

4. **Output format** – one line per task, no list markers: 

Here's an example 

Create a <chart‑type> to show <insight> using <x_column> on the x‑axis and <y_column> on the y‑axis (<aggregation>).

– Replace `<aggregation>` with avg, sum, count, median, etc., or drop it for raw values.  
– If two columns go on the same axis (e.g. grouped bar), mention both.  
– End the sentence with the proposed chart type in parentheses.

5. **Example** (show the style you must produce):  

Create a grouped bar chart to compare average salary_in_usd for each experience_level across company_size (grouped bar chart).

Return exactly four lines that follow rule 4.           

            """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"Given the context and metadata below, generate 4 short data analysis questions:\n{combined_context}"
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
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


# ++++++++++++++++++++++++ Generate Multiple Questions Sub Function +++++++++++++++++++++++++ #

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
            model="gpt-4.1-nano",
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

def infer_category_from_prompt(prompt: str) -> str:
    prompt = prompt.lower()
    if "scatter" in prompt: return "Correlation"
    if "line chart" in prompt: return "Trend"
    if "box" in prompt: return "Outlier"
    if "bar" in prompt and "grouped" in prompt: return "Comparison"
    if "bar" in prompt: return "Distribution"
    if "heatmap" in prompt: return "Correlation"
    return "Meta"


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Expanding Node with Questions Function XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

def expand_root_node(clicked_node):
    themes = [
        ("Distribution", "Explore distributions of numeric columns"),
        ("Comparison", "Compare categories using counts or averages"),
        ("Correlation", "Analyze relationships between numeric features"),
        ("Outlier", "Identify extreme values or unusual groups"),
        ("Trend", "Visualize changes over time or order"),
        ("Relationship", "Explore multi-variable relationships"),
    ]

    parent_path = clicked_node.node_id

    for idx, (category, full_question) in enumerate(themes, start=1):
        child_path = f"{parent_path}.{idx}"
        label = category  # short label
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        node = MindMapNode(
            node_id=child_path,
            label=label,
            full_question=full_question,  # << more informative prompt
            category=category,
            node_type="thematic",
            parent_id=parent_path,
            color=color
        )

        st.session_state.mindmap_nodes[child_path] = node
        st.session_state.curr_state.nodes.append(node.to_streamlit_node())
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(f"{parent_path}-{child_path}", parent_path, child_path, animated=True)
        )

    clicked_node.mark_expanded()
    st.session_state.expanded_nodes.add(parent_path)

    # Optional: log this click for downstream insights
    st.session_state.clicked_questions.append({
        "section": parent_path,
        "short_label": clicked_node.label,
        "full_question": "Root node expanded"
    })


def expand_node_with_questions(clicked_node):
    """
    Expand a thematic node (e.g. Distribution, Trend) into 4 EDA-style analysis questions.
    """
    theme = clicked_node.full_question
    parent_path = clicked_node.node_id

    context = f"Theme: {theme}\n\nDataset metadata:\n{st.session_state.metadata_string}"
    questions = get_list_questions(context)
    short_labels = paraphrase_questions(questions)

    child_paths = get_section_path_children(parent_path, len(questions))

    for i, child_path in enumerate(child_paths):
        full_q = questions[i]
        label = short_labels[i]
        category = infer_category_from_prompt(full_q)
        color = get_color_for_depth(child_path)

        node = MindMapNode(
            node_id=child_path,
            label=label,
            full_question=full_q,
            category=category,
            node_type="exploratory",
            payload="",  # you can extract chart-type/X/Y later
            parent_id=parent_path,
            color=color
        )

        # Register and visualize
        st.session_state.mindmap_nodes[child_path] = node
        st.session_state.curr_state.nodes.append(node.to_streamlit_node())
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(f"{parent_path}-{child_path}", parent_path, child_path, animated=True)
        )

    clicked_node.mark_expanded()
    st.session_state.expanded_nodes.add(parent_path)

    # Optional: log this click
    if parent_path not in {q["section"] for q in st.session_state.clicked_questions}:
        st.session_state.clicked_questions.append({
            "section": parent_path,
            "short_label": clicked_node.label,
            "full_question": clicked_node.full_question
        })


PAGE_OPTIONS = [
    'Data Upload',
    'Mind Mapping',
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
                st.session_state["DATA_RAW"] = df
                st.session_state.df_preview = df.head()
                # st.session_state.df_summary = df.describe()

                # Save dataset name without extension
                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name
                # st.write(dataset_name)


                # numeric + categorical summary
                numeric_summary = df.describe()
                cat_summary = df.describe(include=['object', 'category', 'bool'])
                # build a richer metadata string
                st.session_state.df_summary = numeric_summary  # keep for display
                cols = df.columns.tolist()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                cat_cardinalities = {c: int(df[c].nunique()) for c in categorical_cols}
                top_cats = {c: df[c].value_counts().head(3).to_dict() for c in categorical_cols}

                # Rebuild a 'metadata_string' for the root node
                if st.session_state.df_summary is not None:
                    # Basic example of turning summary + columns into a string
                    cols = list(st.session_state.df_summary.columns)
                    row_count = st.session_state.df.shape[0]
                    st.session_state.metadata_string = (
                        f"Columns: {cols}\n"
                        f"Numeric columns: {numeric_cols}\n"
                        f"Categorical columns: {categorical_cols} (cardinalities: {cat_cardinalities})\n"
                        f"Top categories: {top_cats}\n"
                        f"Row count: {len(df)}"
                    )
                    # print(st.session_state.metadata_string)
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


    elif page == 'Mind Mapping':
        st.title("Mind Mapping + Agentic Ensemble")

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
        clicked_node = st.session_state.mindmap_nodes.get(clicked_node_id)

        if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
            clicked_node = st.session_state.mindmap_nodes.get(clicked_node_id)

            # ✅ check if the node exists in your map
            if clicked_node is not None and clicked_node.can_expand():
                if clicked_node.node_type == "root":
                    expand_root_node(clicked_node)
                else:
                    expand_node_with_questions(clicked_node)
                clicked_node.mark_expanded()
                st.session_state.expanded_nodes.add(clicked_node_id)

            st.rerun()
        # Display a table of all clicked questions so far
        if st.session_state.clicked_questions:
            st.write("## Questions Clicked So Far")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            st.table(df_log)
