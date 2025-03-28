# This app creates a recursive mind map.
# Clicking the center node adds 4 children.
# Clicking any child node adds 4 more children to that node.

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import RadialLayout
from resources.documentation_page_1 import documentation_page

# ----------------- Color Palette ------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# ----------------- Helper Functions ------------------
def get_color_for_depth(depth):
    return COLOR_PALETTE[depth % len(COLOR_PALETTE)]

def get_node_depth(node_id):
    return node_id.count('_')

def add_children_to_node(node_id):
    depth = get_node_depth(node_id) + 1
    color = get_color_for_depth(depth)
    current_children = [n for n in st.session_state.nodes if n.id.startswith(f"{node_id}_")]
    base_count = len(current_children)

    for i in range(1, 5):
        new_id = f"{node_id}_{base_count+i}"
        new_label = f"Node {new_id.split('_')[-1]}"
        new_node = StreamlitFlowNode(
            new_id,
            pos=(i, 0),
            data={"content": new_label},
            node_type="default",
            source_position="right",
            target_position="left",
            style={"backgroundColor": color}
        )
        new_edge = StreamlitFlowEdge(f"{node_id}-{new_id}", node_id, new_id, animated=True)

        st.session_state.nodes.append(new_node)
        st.session_state.edges.append(new_edge)
    st.session_state.expanded_nodes.add(node_id)

# ----------------- Stream Flow Page ------------------
def stream_flow():
    state = StreamlitFlowState(nodes=st.session_state.nodes, edges=st.session_state.edges)
    updated_state = streamlit_flow(
        "mind_map",
        state,
        layout=RadialLayout(),
        fit_view=True,
        height=600,
        get_node_on_click=True
    )

    clicked = updated_state.selected_id
    if clicked and clicked not in st.session_state.expanded_nodes:
        add_children_to_node(clicked)
        st.rerun()

# ----------------- Sidebar Navigation ------------------
PAGE_OPTIONS = [
    "Stream Flow",
    "Documentation"
]

page = st.sidebar.radio("Select a Page", PAGE_OPTIONS)

# ----------------- Session State Initialization ------------------
if "nodes" not in st.session_state:
    root = StreamlitFlowNode("root", (0, 0), {"content": "Dataset"}, "input", "right", style={"backgroundColor": COLOR_PALETTE[0]})
    st.session_state.nodes = [root]
    st.session_state.edges = []
    st.session_state.expanded_nodes = set()

# ----------------- Page Routing ------------------
if page == "Stream Flow":
    st.title("📊 Interactive Data Mind Map")
    st.caption("Click a node to add 4 new branches. Explore recursively.")
    stream_flow()

elif page == "Documentation":
    documentation_page()