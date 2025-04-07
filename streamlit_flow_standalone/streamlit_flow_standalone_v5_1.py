import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout, RadialLayout, ManualLayout
from uuid import uuid4
import random
import openai
import pandas as pd

st.set_page_config("Multi-Flow App", layout="wide")

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a page", ["📁 Data Upload", "🧠 Mind Mapping", "🎨 Flow Editor", "🔍 Inspector"])

# ------------------- Color Setup -------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# ------------------- Session State Setup -------------------
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
if 'dataset_name' not in st.session_state:
    st.session_state['dataset_name'] = ""
# For storing each node the user has clicked (to display full question)
if "clicked_questions" not in st.session_state:
    st.session_state.clicked_questions = []
# For controlling repeated expansions
if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()


def load_data(uploaded_file):
    """Load CSV/Excel into a DataFrame."""
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
# --------------------------------------------------------------------------------
# OPENAI Question Generation + Identification
# --------------------------------------------------------------------------------
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
            specific data relationships or trends, referencing the relevant columns when possible."""
        },
        {
            "role": "user",
            "content": f"Given the context and metadata below, generate 5 short data analysis questions:\n{combined_context}"
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
               bullet points, or additional text."""
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



# --------------------------------------------------------------------------------
# Node/Flow Utils
# --------------------------------------------------------------------------------
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
                "content": child_short_label
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
    # Hard-coded list of the top-level EDA themes
    themes = [
        ("Distributions", "#FF6B6B"),
        ("Correlations", "#6BCB77"),
        ("Missingness", "#4D96FF"),
        ("Data Types", "#FFD93D")
    ]

    parent_path = clicked_node.data["section_path"]  # "S0"

    for i, (theme_label, color) in enumerate(themes):
        # Example: child_id = "S0.1" for the first theme
        child_id = f"{parent_path}.{i + 1}"

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

    # Log the parent's question if not already
    parent_section = clicked_node.data["section_path"]
    existing_paths = [q["section"] for q in st.session_state.clicked_questions]
    if parent_section not in existing_paths:
        st.session_state.clicked_questions.append({
            "section": parent_section,
            "short_label": clicked_node.data.get("short_label", "ROOT"),
            "full_question": "Root node expanded",
            "node_type": clicked_node.data.get("node_type", "unknown")
        })
# --------------------------------------------------------------------------------
# Initialize Root Node/Flow State
# --------------------------------------------------------------------------------
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
            "content": dataset_label,
            "node_type": "root",
        },
        "input",
        "right",
        style={"backgroundColor": COLOR_PALETTE[0]}
    )
    st.session_state.curr_state = StreamlitFlowState(nodes=[root_node], edges=[])
    st.session_state.expanded_nodes = set()

# --------------------------------------------------------------------------------
# Page: Data Upload
# --------------------------------------------------------------------------------
if page == '📁 Data Upload':
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



# --------------------------------------------------------------------------------
# Page: Mind Mapping
# --------------------------------------------------------------------------------
elif page == "🧠 Mind Mapping":
    st.title('Mind Mapping + OpenAI Ensemble')

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
        show_minimap=True
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

# --------------------------------------------------------------------------------
# Page: Flow Editor
# --------------------------------------------------------------------------------
elif page == "🎨 Flow Editor":
    st.title("🎨 Flow Editor")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("➕ Add Node"):
            new_id = str(uuid4())
            new_node = StreamlitFlowNode(
                new_id,
                (random.randint(0, 5), random.randint(0, 5)),
                {
                    "content": f'Node {len(st.session_state.curr_state.nodes) + 1}',
                    "section_path": new_id,
                    "full_question": "Manually added node"
                },
                'default', 'right', 'left'
            )
            st.session_state.curr_state.nodes.append(new_node)
            st.rerun()

    with col2:
        if st.button("➖ Delete Random Node"):
            if st.session_state.curr_state.nodes:
                node = random.choice(st.session_state.curr_state.nodes)
                st.session_state.curr_state.nodes = [
                    n for n in st.session_state.curr_state.nodes if n.id != node.id
                ]
                st.session_state.curr_state.edges = [
                    e for e in st.session_state.curr_state.edges
                    if e.source != node.id and e.target != node.id
                ]
                st.rerun()

    with col3:
        if st.button("🔗 Add Random Edge"):
            nodes = st.session_state.curr_state.nodes
            if len(nodes) > 1:
                s = random.choice(nodes)
                t = random.choice(nodes)
                if s.id != t.id:
                    new_edge = StreamlitFlowEdge(f"{s.id}-{t.id}", s.id, t.id, animated=True)
                    if not any(e.id == new_edge.id for e in st.session_state.curr_state.edges):
                        st.session_state.curr_state.edges.append(new_edge)
                        st.rerun()

    with col4:
        if st.button("❌ Delete Random Edge"):
            if st.session_state.curr_state.edges:
                edge = random.choice(st.session_state.curr_state.edges)
                st.session_state.curr_state.edges = [
                    e for e in st.session_state.curr_state.edges if e.id != edge.id
                ]
                st.rerun()

    with col5:
        if st.button("🎲 Random Flow"):
            nodes = []
            edges = []
            for i in range(5):
                node_id = str(uuid4())
                nodes.append(StreamlitFlowNode(
                    node_id, (i, i),
                    {
                        "content": f"Node {i}",
                        "section_path": f"S{i}",
                        "full_question": f"Random question {i}"
                    },
                    "default", "right", "left"
                ))
            for i in range(4):
                edges.append(StreamlitFlowEdge(
                    f"{nodes[i].id}-{nodes[i + 1].id}",
                    nodes[i].id,
                    nodes[i + 1].id,
                    animated=True
                ))

            st.session_state.curr_state = StreamlitFlowState(nodes, edges)
            st.session_state.expanded_nodes = set()
            st.rerun()

    st.session_state.curr_state = streamlit_flow(
        "editor_flow",
        st.session_state.curr_state,
        # layout=TreeLayout(direction="right"),
        fit_view=True,
        height=550,
        enable_node_menu=True,
        enable_edge_menu=True,
        enable_pane_menu=True,
        allow_new_edges=True,
        get_node_on_click=True,
        get_edge_on_click=True,
        show_minimap=True
    )

# --------------------------------------------------------------------------------
# Page: Inspector
# --------------------------------------------------------------------------------
elif page == "🔍 Inspector":
    st.title("🔍 Flow Inspector")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🧩 Nodes")
        for node in st.session_state.curr_state.nodes:
            st.json(vars(node))

    with col2:
        st.subheader("🔗 Edges")
        for edge in st.session_state.curr_state.edges:
            st.json(vars(edge))

    with col3:
        st.subheader("🖱️ Selected ID")
        st.write(st.session_state.curr_state.selected_id)
