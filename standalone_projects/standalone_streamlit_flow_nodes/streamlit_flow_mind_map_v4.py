import pandas as pd
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import random
import numpy as np
# from standalone_projects.standalone_streamlit_flow_nodes.mindmap_config import sample_categories, CATEGORY_CFG
import openai
from mindmap_config import CATEGORY_CFG, sample_categories
import re
from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

if "curr_state" not in st.session_state:
    # Prepare root node. We'll store the dataset metadata in "full_question" if we have it.
    dataset_label = st.session_state.get("dataset_name", "Dataset")

    root_theme = ThemeNode(
        node_id="S0",
        label="ROOT",
        full_question="Overview of the dataset",
        category="Meta",
        node_type="theme",
        parent_id=None,
        color=COLOR_PALETTE[0],
        metadata={"content": dataset_label}
    )

    st.session_state.mindmap_nodes = {"S0": root_theme}
    st.session_state.curr_state = StreamlitFlowState(
        nodes=[root_theme.to_streamlit_node()],
        edges=[]
    )

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index", "clicked_questions", "dataset_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "DATA_RAW"] else []

# Set type for expanded_nodes properly
if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()


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


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Expanding Node with Questions Function XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

def expand_root_node(clicked_node):
    """
    Expand the ROOT node into EDA archetype themes:
    Histogram, Bar Chart, Scatter Plot, Box Plot, Heatmap, Pivot Table.
    """
    themes = [
        ("Histogram", "#FF6B6B"),
        ("Bar Chart", "#6BCB77"),
        ("Scatter Plot", "#4D96FF"),
        ("Box Plot", "#FFD93D"),
        ("Heatmap", "#845EC2"),
        ("Pivot Table", "#F9A826"),
    ]

    parent_path = clicked_node.data["section_path"]

    # 1) Create one child node per theme
    for idx, (theme_label, color) in enumerate(themes, start=1):
        child_path = f"{parent_path}.{idx}"
        node_data = {
            "section_path": child_path,
            "short_label": theme_label,
            "full_question": theme_label,  # use the theme as the query context
            "content": f"**{theme_label}**",
            "node_type": "thematic"
        }

        new_node = StreamlitFlowNode(
            child_path,
            (random.randint(-100, 100), random.randint(-100, 100)),
            node_data,
            "default",  # node shape/type
            "right",  # target handle position
            "left",  # source handle position
            style={"backgroundColor": color}
        )
        st.session_state.curr_state.nodes.append(new_node)

        edge_id = f"{clicked_node.id}-{child_path}"
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(edge_id, clicked_node.id, child_path, animated=True)
        )

    # 2) Mark this node as expanded
    st.session_state.expanded_nodes.add(clicked_node.id)

    # 3) Log that we've expanded the ROOT (once)
    parent_section = parent_path
    existing = [q["section"] for q in st.session_state.clicked_questions]
    if parent_section not in existing:
        st.session_state.clicked_questions.append({
            "section": parent_section,
            "short_label": clicked_node.data.get("short_label", "ROOT"),
            "full_question": "Root node expanded"
        })


def expand_node_with_questions(clicked_node):
    """
    Expand any thematic node into 4 EDA‑style questions that
    consider both numeric and categorical columns.
    """
    # 1) Determine theme (e.g. "Histogram", "Bar Chart", etc.)
    theme = clicked_node.data.get("full_question", "")

    context = f"Theme: {theme}\n\nDataset metadata:\n{st.session_state.metadata_string}"
    q1 = get_list_questions(context)
    short_labels = paraphrase_questions(q1)

    # 5) Create child nodes for each question
    parent_path = clicked_node.data.get("section_path", "S0")
    child_paths = get_section_path_children(parent_path, num_children=len(q1))
    for i, child_path in enumerate(child_paths):
        full_q = q1[i]
        label = short_labels[i]
        color = get_color_for_depth(child_path)

        node_data = {
            "section_path": child_path,
            "short_label": label,
            "full_question": full_q,
            "content": f"**{label}**"
        }

        new_node = StreamlitFlowNode(
            child_path,
            (random.randint(-100, 100), random.randint(-100, 100)),
            node_data,
            "default",
            "right",
            "left",
            style={"backgroundColor": color}
        )
        st.session_state.curr_state.nodes.append(new_node)
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(f"{clicked_node.id}-{child_path}", clicked_node.id, child_path, animated=True)
        )

    # 6) Mark as expanded & log the click
    st.session_state.expanded_nodes.add(clicked_node.data["section_path"])
    if parent_path not in {q["section"] for q in st.session_state.clicked_questions}:
        st.session_state.clicked_questions.append({
            "section": parent_path,
            "short_label": clicked_node.data.get("short_label", parent_path),
            "full_question": theme
        })

PAGE_OPTIONS = [
    'Data Upload',
    'Mind Mapping',
    'Data Analyst',
    'Data Storytelling'
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

        # ------------------ Mind Map Node Click Event Handler ------------------

        clicked_node_id = st.session_state.curr_state.selected_id

        if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
            st.write("👉 Node clicked:", clicked_node_id)

            # Get the corresponding object from our mindmap node registry
            clicked_obj = st.session_state.mindmap_nodes.get(clicked_node_id)

            # Log for debugging
            st.write("📌 All registered node IDs:", list(st.session_state.mindmap_nodes.keys()))
            st.write("📌 Already expanded nodes:", st.session_state.expanded_nodes)

            if not clicked_obj:
                st.warning(f"⚠️ Node '{clicked_node_id}' not found in mindmap_nodes.")
            elif not clicked_obj.can_expand():
                st.info(f"ℹ️ Node '{clicked_node_id}' is a terminal node or already expanded.")
            else:
                # ✅ Generate children using the class-specific logic
                children = clicked_obj.get_children(
                    openai_client=client,
                    metadata_string=st.session_state.metadata_string
                )

                for child in children:
                    st.session_state.mindmap_nodes[child.node_id] = child  # 🔑 Register it to mindmap dict
                    st.session_state.curr_state.nodes.append(child.to_streamlit_node())
                    st.session_state.curr_state.edges.append(
                        StreamlitFlowEdge(
                            f"{clicked_obj.node_id}-{child.node_id}",
                            clicked_obj.node_id,
                            child.node_id,
                            animated=True
                        )
                    )

                # ✅ Mark this as expanded
                clicked_obj.mark_expanded()
                st.session_state.expanded_nodes.add(clicked_node_id)

                # ✅ Log the click
                st.session_state.clicked_questions.append({
                    "section": clicked_obj.node_id,
                    "short_label": clicked_obj.label,
                    "full_question": clicked_obj.full_question,
                    "node_type": clicked_obj.node_type
                })

                # 🔁 Re-render
                st.rerun()

        # ------------------ Display User Click History ------------------

        if st.session_state.clicked_questions:
            st.write("## Clicked Nodes (User's Exploration Path)")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            df_log = df_log[["section", "short_label", "node_type", "full_question"]]
            st.table(df_log)