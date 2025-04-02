import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import pandas as pd
from uuid import uuid4
from openai import OpenAI
import secrets

# ------------------------------
# Config & API Init
# ------------------------------
st.set_page_config("Streamlit Flow: Dataset Mind Map", layout="wide")
st.title("📊 AI Dataset Mind Mapping")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------
# Question Generator
# ------------------------------
def generate_questions_from_metadata():
    df = st.session_state.df
    metadata = {
        "columns": df.columns.tolist(),
        "summary": st.session_state.df_summary.to_dict(),
        "row_count": len(df)
    }
    metadata_string = f"Columns: {', '.join(metadata['columns'])}\nTotal Rows: {metadata['row_count']}\nSummary Stats: {metadata['summary']}"

    messages = [
        {"role": "system", "content": "You're a data analyst generating concise, insightful questions for data exploration."},
        {"role": "user", "content": f"Given this metadata:\n\n{metadata_string}\n\nGenerate 5 questions suitable for data analysis."}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        questions = [q.strip().lstrip("12345. ") for q in response.choices[0].message.content.split("\n") if q.strip()]
        return questions[:5]  # Cap at 5
    except Exception as e:
        return [f"Error: {e}"]

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

if uploaded_file and "df_loaded" not in st.session_state:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.session_state.df = df
    st.session_state.df_summary = df.describe()
    st.session_state.df_preview = df.head()
    st.session_state.df_name = uploaded_file.name.split('.')[0]
    st.session_state.df_loaded = True

    st.success("Dataset uploaded! Now click 'Generate Questions' to build your mind map.")

# ------------------------------
# Button to Generate Flow
# ------------------------------
if st.session_state.get("df_loaded") and not st.session_state.get("flow_generated"):
    if st.button("🧠 Generate Questions & Create Mind Map"):
        questions = generate_questions_from_metadata()

        root_node = StreamlitFlowNode("root", (0, 0), {'content': st.session_state.df_name}, 'input', 'right')
        question_nodes = []
        edges = []

        for i, question in enumerate(questions):
            node_id = f"q{i+1}"
            q_node = StreamlitFlowNode(node_id, (i+1, 0), {'content': question}, 'default', 'right', 'left')
            question_nodes.append(q_node)
            edges.append(StreamlitFlowEdge(f"root-{node_id}", "root", node_id, animated=True, marker_end={'type': 'arrow'}))

        st.session_state.curr_state = StreamlitFlowState(nodes=[root_node] + question_nodes, edges=edges)
        st.session_state.flow_generated = True
        st.rerun()

# ------------------------------
# Preview Table
# ------------------------------
if "df_preview" in st.session_state:
    st.write("### Data Preview")
    st.dataframe(st.session_state.df_preview)
    st.write("### Summary")
    st.dataframe(st.session_state.df_summary)

# ------------------------------
# Fallback Flow
# ------------------------------
if 'curr_state' not in st.session_state:
    nodes = [
        StreamlitFlowNode("1", (0, 0), {'content': 'Sample Dataset'}, 'input', 'right'),
        StreamlitFlowNode("2", (1, 0), {'content': 'Any outliers in the data?'}, 'default', 'right', 'left')
    ]
    edges = [StreamlitFlowEdge("1-2", "1", "2", animated=True, marker_end={'type': 'arrow'})]
    st.session_state.curr_state = StreamlitFlowState(nodes, edges)

# ------------------------------
# Optional Flow UI Buttons
# ------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("➕ Add Node"):
        new_node = StreamlitFlowNode(str(uuid4()), (0, 0), {'content': 'New Node'}, 'default', 'right', 'left')
        st.session_state.curr_state.nodes.append(new_node)
        st.rerun()

with col2:
    if st.button("❌ Delete Random Node"):
        if len(st.session_state.curr_state.nodes) > 1:
            node_to_delete = secrets.choice([n for n in st.session_state.curr_state.nodes if n.id != "root"])
            st.session_state.curr_state.nodes = [n for n in st.session_state.curr_state.nodes if n.id != node_to_delete.id]
            st.session_state.curr_state.edges = [e for e in st.session_state.curr_state.edges if e.source != node_to_delete.id and e.target != node_to_delete.id]
            st.rerun()

with col3:
    if st.button("🔗 Add Random Edge"):
        nodes = st.session_state.curr_state.nodes
        if len(nodes) > 1:
            src = secrets.choice(nodes)
            tgt = secrets.choice(nodes)
            if src.id != tgt.id:
                edge_id = f"{src.id}-{tgt.id}"
                if all(e.id != edge_id for e in st.session_state.curr_state.edges):
                    st.session_state.curr_state.edges.append(StreamlitFlowEdge(edge_id, src.id, tgt.id, animated=True))
                    st.rerun()

with col4:
    if st.button("✂️ Delete Random Edge"):
        if len(st.session_state.curr_state.edges) > 0:
            edge = secrets.choice(st.session_state.curr_state.edges)
            st.session_state.curr_state.edges = [e for e in st.session_state.curr_state.edges if e.id != edge.id]
            st.rerun()

with col5:
    if st.button("🎲 Random Flow"):
        nodes = [StreamlitFlowNode(str(uuid4()), (0, 0), {'content': f'Node {i}'}, 'default', 'right', 'left') for i in range(5)]
        edges = []
        for _ in range(4):
            src, tgt = secrets.SystemRandom().sample(nodes, 2)
            edge_id = f"{src.id}-{tgt.id}"
            edges.append(StreamlitFlowEdge(edge_id, src.id, tgt.id, animated=True))
        st.session_state.curr_state = StreamlitFlowState(nodes, edges)
        st.rerun()

# ------------------------------
# Render Flow
# ------------------------------
st.session_state.curr_state = streamlit_flow(
    'example_flow',
    st.session_state.curr_state,
    layout=TreeLayout(direction='right'),
    fit_view=True,
    height=500,
    enable_node_menu=True,
    enable_edge_menu=True,
    enable_pane_menu=True,
    get_edge_on_click=True,
    get_node_on_click=True,
    show_minimap=True,
    hide_watermark=True,
    allow_new_edges=True,
    min_zoom=0.1
)

# ------------------------------
# Debug Panels
# ------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📌 Nodes")
    for node in st.session_state.curr_state.nodes:
        st.write(node)
with col2:
    st.markdown("### 🔗 Edges")
    for edge in st.session_state.curr_state.edges:
        st.write(edge)
with col3:
    st.markdown("### 🔍 Selected")
    st.write(st.session_state.curr_state.selected_id)
