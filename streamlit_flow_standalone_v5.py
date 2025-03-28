import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout, RadialLayout, ManualLayout
from uuid import uuid4
import random

st.set_page_config("Multi-Flow App", layout="wide")

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a page", ["🧠 Mind Mapping", "🎨 Flow Editor", "🔍 Inspector"])

# ------------------- Color Setup -------------------
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

def get_color_for_depth(depth):
    return COLOR_PALETTE[depth % len(COLOR_PALETTE)]

def get_node_depth(node_id):
    return node_id.count("_")  # each underscore = one level deeper

# ------------------- Initialize Session -------------------
if "curr_state" not in st.session_state:
    root = StreamlitFlowNode("root", (0, 0), {"content": "Dataset"}, "input", "right", style={"backgroundColor": "#FF6B6B"})
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
if page == "🧠 Mind Mapping":
    st.title("🧠 Recursive Mind Map Explorer")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Reset Mind Map"):
            root = StreamlitFlowNode("root", (0, 0), {"content": "Dataset"}, "input", "right", style={"backgroundColor": "#FF6B6B"})
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
