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
from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# ##################### SESSION STATE STUFF ####################### #

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

if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()


# ######################## UPLOAD DATA ######################## #

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

# ######################## MIND MAPPING ######################## #

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
            else:
                # ✅ Only log if not already recorded
                already_logged = any(q["section"] == clicked_obj.node_id for q in st.session_state.clicked_questions)

                if not already_logged:
                    st.session_state.clicked_questions.append({
                        "section": clicked_obj.node_id,
                        "short_label": clicked_obj.label,
                        "full_question": clicked_obj.full_question,
                        "node_type": clicked_obj.node_type
                    })

                if clicked_obj.can_expand() and clicked_node_id not in st.session_state.expanded_nodes:
                    children = clicked_obj.get_children(
                        openai_client=client,
                        metadata_string=st.session_state.metadata_string
                    ) or []

                    for child in children:
                        st.session_state.mindmap_nodes[child.node_id] = child
                        st.session_state.curr_state.nodes.append(child.to_streamlit_node())
                        st.session_state.curr_state.edges.append(
                            StreamlitFlowEdge(
                                f"{clicked_obj.node_id}-{child.node_id}",
                                clicked_obj.node_id,
                                child.node_id,
                                animated=True
                            )
                        )

                    clicked_obj.mark_expanded()
                    st.session_state.expanded_nodes.add(clicked_node_id)

                st.rerun()

        # ------------------ Display User Click History ------------------

        if st.session_state.clicked_questions:
            st.write("## Clicked Nodes (User's Exploration Path)")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            df_log = df_log[["section", "short_label", "node_type", "full_question"]]
            st.table(df_log)