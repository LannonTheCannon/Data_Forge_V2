# This app creates a recursive mind map.
# Clicking the center node adds 4 children.
# Clicking any child node adds 4 more children to that node.

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import RadialLayout
from uuid import uuid4

st.set_page_config("Recursive Mind Map", layout="wide")
st.title("🧠 Recursive Mind Map Explorer")

# ----------------- Init Session ------------------
if "curr_state" not in st.session_state:
    root = StreamlitFlowNode("root", (0, 0), {"content": "Dataset"}, "input", "right", style={"backgroundColor": "#FF6B6B"})
    st.session_state.curr_state = StreamlitFlowState(nodes=[root], edges=[])
    st.session_state.expanded_nodes = set()
    st.session_state.color_map = {}  # node_id -> color

# ----------------- Color Palette ------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

def get_color_for_depth(depth):
    return COLOR_PALETTE[depth % len(COLOR_PALETTE)]

def get_node_depth(node_id):
    return node_id.count("_")  # each underscore = one level deeper

# ----------------- Add Children Function ------------------
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

# ----------------- Render Flow ------------------
st.session_state.curr_state = streamlit_flow(
    "recursive_map",
    st.session_state.curr_state,
    layout=RadialLayout(),
    fit_view=True,
    height=500,
    get_node_on_click=True,
    enable_node_menu=True,
    enable_edge_menu=True
)

clicked_node_id = st.session_state.curr_state.selected_id

if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
    add_children(clicked_node_id)
    st.rerun()

# ----------------- Sidebar ------------------
st.sidebar.markdown("### Instructions")
st.sidebar.write("- Click any node to expand it with 4 children")
st.sidebar.write("- Expansions are recursive and color-coded by depth")

PAGE_OPTIONS = [
    'Mind Mapping',
    'Hello World'
]

page = st.sidebar.radio("Go to", ["Flow Editor", "Inspector"])

if page == 'Mind Mapping':
    root = StreamlitFlowNode("root", (0, 0), {"content": "Dataset"}, "input", "right", style={"backgroundColor": "#FF6B6B"})
    st.session_state.curr_state = StreamlitFlowState(nodes=[root], edges=[])
    st.session_state.expanded_nodes = set()
    st.session_state.color_map = {}
    st.rerun()

elif page == 'Hello World':
    pass