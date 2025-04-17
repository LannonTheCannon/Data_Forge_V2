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

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index", "clicked_questions", "dataset_name", "expanded_nodes"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string",
                                                "DATA_RAW"] else []


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

# ######################### Mind Mappping function ######################### #

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

# ++++++++++++++++++++++++ Generate Multiple Questions Sub Function +++++++++++++++++++++++++ #

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

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Expanding Node with Questions Function XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

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
        child_id = f'{parent_path}.{i+1}'
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

# ######################### Data Charting and Table Viz function ######################### #

def get_assistant_interpretation(user_input, metadata, valid_columns):
    column_names = ', '.join(valid_columns)

    prompt = f"""
You are a visualization interpreter.

Your job is to rephrase the user's request into a **precise and code-compatible** instruction. Use this format:

→ "Create a [chart type] of the `[y_column]` on the y-axis ([aggregation]) and the `[x_column]` in the x-axis and make the chart [color]."

---

Rules:
- DO NOT invent or guess column names. Use ONLY from this list:
  {column_names}
- NEVER say "average salary in USD" — instead say: "`salary_in_usd` on the y-axis (avg)"
- Keep aggregation words like "avg", "sum", or "count" OUTSIDE of the column name.
- Keep axis mappings clear and exact.
- Mention the color explicitly at the end.
- Avoid words like “visualize” or “illustrate.” Just say "Create a bar chart..."

---

📥 USER QUERY:
{user_input}

📊 METADATA:
{metadata}

✍️ Respond with just one sentence using the format shown above.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites data visualization queries into precise and code-friendly instructions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        st.warning(f"Error in get_assistant_interpretation: {e}")
        return "Could not interpret user request."

def display_chat_history():
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}

    for i, msg in enumerate(msgs.messages):
        role_label = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role_label}:** {msg.content}")

            if i in st.session_state["chat_artifacts"]:
                for j, artifact in enumerate(st.session_state["chat_artifacts"][i]):
                    unique_key = f"msg_{i}_artifact_{j}"
                    editor_key = f"editor_code_{unique_key}"
                    output_key = f"output_chart_{unique_key}"

                    with st.expander(f"\U0001F4CE {artifact['title']}", expanded=True):
                        tabs = st.tabs(["📊 Output", "💻 Code"])

                        # --- Code Tab First, to capture edits and trigger updates ---
                        with tabs[1]:
                            code_before = st.session_state.get(editor_key, artifact.get("code", ""))
                            editor_response = code_editor(
                                code=code_before,
                                lang="python",
                                theme="dracula",
                                height=300,
                                buttons=[
                                    {
                                        "name": "Run",
                                        "feather": "Play",
                                        "primary": True,
                                        "hasText": True,
                                        "showWithIcon": True,
                                        "commands": ["submit"],
                                        "style": {"bottom": "0.44rem", "right": "0.4rem"}
                                    }
                                ],
                                key=f"code_editor_{unique_key}"
                            )

                            new_code = editor_response.get("text", "").strip()

                            # Only run if the code has changed
                            if new_code and new_code != st.session_state.get(editor_key):
                                try:
                                    exec_globals = {
                                        "df": st.session_state.df,
                                        "pd": pd,
                                        "np": np,
                                        "sns": sns,
                                        "go": go,
                                        "plt": plt,
                                        "pio": pio,
                                        "st": st,
                                        "json": json
                                    }
                                    exec_locals = {}
                                    exec(new_code, exec_globals, exec_locals)

                                    output_obj = exec_locals.get("fig") or \
                                                 exec_locals.get("output") or \
                                                 exec_locals.get("fig_dict")

                                    if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                                        output_obj = pio.from_json(json.dumps(output_obj))

                                    artifact["data"] = output_obj
                                    artifact["render_type"] = "plotly" if isinstance(output_obj, go.Figure) else "dataframe"
                                    st.session_state[editor_key] = new_code
                                    st.session_state[output_key] = output_obj

                                except Exception as e:
                                    st.error(f"Error executing code: {e}")

                        # --- Output Tab ---
                        with tabs[0]:
                            output_obj = st.session_state.get(output_key, artifact.get("data"))
                            render_type = artifact.get("render_type")

                            if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                                output_obj = pio.from_json(json.dumps(output_obj))

                            if render_type == "plotly":
                                st.plotly_chart(
                                    output_obj,
                                    use_container_width=True,
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "displaylogo": False
                                    },
                                    key=f"plotly_{output_key}"
                                )
                            elif render_type == "dataframe":
                                st.dataframe(output_obj, key=f"df_{output_key}")
                            else:
                                st.write(output_obj)

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

    elif page == 'Data Analyst':
        st.subheader('Pandas Data Analyst Mode')
        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("IMPORTANT: For best results use this formula -> Create a [chart] of the [field] on the y-axis (aggregation) and the [field] on the x-axis and make the chart [color].")
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4.1-nano', api_key=st.secrets['OPENAI_API_KEY'])
            st.session_state.pandas_data_analyst = PandasDataAnalyst(
                model=model,
                data_wrangling_agent=DataWranglingAgent(model=model,
                                                        log=True,
                                                        n_samples=100),
                data_visualization_agent=DataVisualizationAgent(
                                                        model=model,
                                                        log=True,
                                                        log_path="logs",
                                                        overwrite=False,  # ✅ Ensures every chart gets a separate file
                                                        n_samples=100,
                                                        bypass_recommended_steps=False)
            )
        question = st.chat_input('Ask a question about your dataset!')
        interpretation = get_assistant_interpretation(
            question,
            st.session_state['metadata_string'],
            st.session_state.df.columns  # ✅ pass real column names
        )
        # print(interpretation)

        if question:
            msgs.add_user_message(interpretation)
            with st.spinner("Thinking..."):
                try:
                    st.session_state.pandas_data_analyst.invoke_agent(
                        user_instructions=interpretation,
                        data_raw=st.session_state["DATA_RAW"]
                    )
                    result = st.session_state.pandas_data_analyst.get_response()
                    route = result.get("routing_preprocessor_decision", "")
                    ai_msg = "Here's what I found:"
                    msgs.add_ai_message(ai_msg)
                    msg_index = len(msgs.messages) - 1
                    if "chat_artifacts" not in st.session_state:
                        st.session_state["chat_artifacts"] = {}
                    st.session_state["chat_artifacts"][msg_index] = []
                    if route == "chart" and not result.get("plotly_error", False):
                        plot_obj = pio.from_json(json.dumps(result["plotly_graph"]))
                        st.session_state.plots.append(plot_obj)
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        viz_code = result.get('data_visualization_function', "")
                        wrangle_code = result.get('data_wrangler_function', "")

                        # Combine both functions into one code block
                        combined_code = f"{wrangle_code}\n\n{viz_code}\n\n# Runtime Execution\noutput = data_visualization(data_wrangler([df]))"

                        st.session_state["chat_artifacts"][msg_index].append({
                            "title": "Chart",
                            "render_type": "plotly",
                            "data": plot_obj,
                            "code": combined_code
                        })
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        print(result['data_visualization_function'])

                    elif route == "table":
                        df = result.get("data_wrangled")
                        if df is not None:
                            st.session_state.dataframes.append(df)
                            st.session_state["chat_artifacts"][msg_index].append({
                                "title": "Table",
                                "render_type": "dataframe",
                                "data": df,
                                'code': result.get('data_wrangler_function')
                            })
                except Exception as e:
                    error_msg = f"Error: {e}"
                    msgs.add_ai_message(error_msg)
        display_chat_history()

    elif page == 'Data Storytelling':
        pass