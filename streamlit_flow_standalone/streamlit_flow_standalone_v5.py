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
page = st.sidebar.radio("Choose a page", ["📁 Data Upload","🧠 Mind Mapping", "🎨 Flow Editor", "🔍 Inspector"])

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

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

def generate_multiple_question_sets():
    question_set_1 = get_list_questions()
    question_set_2 = get_list_questions()
    question_set_3 = get_list_questions()

    return question_set_1, question_set_2, question_set_3

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

def get_color_for_depth(depth):
    return COLOR_PALETTE[depth % len(COLOR_PALETTE)]

def get_node_depth(node_id):
    return node_id.count("_")  # each underscore = one level deeper

# ------------------- Initialize Session -------------------
if "curr_state" not in st.session_state:
    dataset_label = st.session_state.get("dataset_name", "Dataset")
    root = StreamlitFlowNode("root", (0, 0), {"content": dataset_label}, "input", "right", style={"backgroundColor": "#FF6B6B"})
    st.session_state.curr_state = StreamlitFlowState(nodes=[root], edges=[])
    st.session_state.expanded_nodes = set()
    st.session_state.color_map = {}

# ------------------- Mind Mapping Logic -------------------
def add_children(parent_id):
    depth = get_node_depth(parent_id) + 1
    color = get_color_for_depth(depth)
    count = sum(1 for n in st.session_state.curr_state.nodes if n.id.startswith(parent_id + "_"))
    new_nodes = []
    new_edges = []
    for i in range(1, 5):
        node_id = f"{parent_id}_{count+i}"
        content = f"Node {node_id.split('_')[-1]}"
        new_nodes.append(StreamlitFlowNode(
            node_id,
            (i, 0),
            {"content": content},
            "default", "right", "left",
            style={"backgroundColor": color}
        ))
        new_edges.append(StreamlitFlowEdge(f"{parent_id}-{node_id}", parent_id, node_id, animated=True))
        st.session_state.color_map[node_id] = color
    st.session_state.curr_state.nodes.extend(new_nodes)
    st.session_state.curr_state.edges.extend(new_edges)
    st.session_state.expanded_nodes.add(parent_id)

# ------------------- Mind Mapping View -------------------
if page == '📁 Data Upload':
    st.title('Upload your own Dataset!')
    uploaded_file = st.file_uploader('Upload CSV and Excel Here', type=['csv', 'excel'])

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

    # Display preview & summary if data exists
    if st.session_state.df is not None:
        st.write("### Data Preview")
        st.write(st.session_state.df_preview)

        st.write("### Data Summary")
        st.write(st.session_state.df_summary)

elif page == "🧠 Mind Mapping":

    # Check if dataset name is present and current root label is outdated
    if st.session_state.get("dataset_name") and st.session_state.curr_state.nodes[0].data["content"] != \
            st.session_state["dataset_name"]:
        st.session_state.curr_state.nodes[0].data["content"] = st.session_state["dataset_name"]

    st.title('Mind Mapping + OpenAI Ensemble Completions')
    st.write('Identify a category of the dataset you would like to explore')

    # Suggested Questions
    st.write('### Suggested Questions')

    if 'question_list' not in st.session_state:
        st.session_state['question_list'] = []

    if st.button('Generate Most Relevant Questions'):
        st.write('Generating multiple question sets...')

        question_set_1, question_set_2, question_set_3 = generate_multiple_question_sets()
        common_questions = identify_common_questions(question_set_1, question_set_2, question_set_3)

        st.session_state["question_list"] = common_questions

    st.write('### Most Relevant Questions')

    for idx, question in enumerate(st.session_state["question_list"]):
        # Okay at this point we need to do some list/ string manipulation
        if (question[0]) in ['1', '2', '3', '4', '5']:
            if st.button(f" 🔍 {question}"):
                #reset_session_variables()
                st.session_state["user_query"] = question
                st.session_state["trigger_assistant"] = True  # Ensure assistant runs
        # else:
        #     st.write(f"{question}")

    Q_list = []
    for idx, question in enumerate(st.session_state['question_list']):
        if question[0] in ['1','2','3','4','5']:
            Q_list.append(question[0])

    for i in Q_list:
        print(i)

    # Now I need to make the nodes into Q_list

    st.title("🧠 Recursive Mind Map Explorer")
    #dataset_label = st.session_state.get('dataset_name', 'Dataset')
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Reset Mind Map"):
            root = StreamlitFlowNode("root",
                                     (0, 0),
                                     {"content": st.session_state.dataset_name},
                                     "input",
                                     "right", style={"backgroundColor": "#FF6B6B"})
            st.session_state.curr_state = StreamlitFlowState(nodes=[root], edges=[])
            st.session_state.expanded_nodes = set()
            st.session_state.color_map = {}
            st.rerun()

    # Render recursive flow
    st.session_state.curr_state = streamlit_flow(
        "mind_map",
        st.session_state.curr_state,
        layout=RadialLayout(),
        fit_view=True,
        height=550,
        get_node_on_click=True,
        enable_node_menu=True,
        enable_edge_menu=True,
        show_minimap=True
    )

    clicked_node_id = st.session_state.curr_state.selected_id
    if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
        add_children(clicked_node_id)
        st.rerun()

# ------------------- Flow Editor View -------------------
elif page == "🎨 Flow Editor":
    st.title("🎨 Flow Editor")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("➕ Add Node"):
            new_node = StreamlitFlowNode(
                str(uuid4()), (random.randint(0, 5), random.randint(0, 5)),
                {'content': f'Node {len(st.session_state.curr_state.nodes)+1}'},
                'default', 'right', 'left'
            )
            st.session_state.curr_state.nodes.append(new_node)
            st.rerun()

    with col2:
        if st.button("➖ Delete Random Node"):
            if st.session_state.curr_state.nodes:
                node = random.choice(st.session_state.curr_state.nodes)
                st.session_state.curr_state.nodes = [n for n in st.session_state.curr_state.nodes if n.id != node.id]
                st.session_state.curr_state.edges = [e for e in st.session_state.curr_state.edges if e.source != node.id and e.target != node.id]
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
                st.session_state.curr_state.edges = [e for e in st.session_state.curr_state.edges if e.id != edge.id]
                st.rerun()

    with col5:
        if st.button("🎲 Random Flow"):
            nodes = [
                StreamlitFlowNode(str(uuid4()), (i, i), {"content": f"Node {i}"}, "default", "right", "left")
                for i in range(5)
            ]
            edges = []
            for i in range(4):
                edges.append(StreamlitFlowEdge(f"{nodes[i].id}-{nodes[i+1].id}", nodes[i].id, nodes[i+1].id, animated=True))
            st.session_state.curr_state = StreamlitFlowState(nodes, edges)
            st.session_state.expanded_nodes = set()
            st.rerun()

    # Render editor layout
    st.session_state.curr_state = streamlit_flow(
        "editor_flow",
        st.session_state.curr_state,
        layout=TreeLayout(direction="right"),
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

# ------------------- Inspector View -------------------
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
