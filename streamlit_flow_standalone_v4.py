import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout  # Or TreeLayout, etc.
import random
from uuid import uuid4

st.set_page_config(layout="wide")
page = st.sidebar.radio("Go to", ["Flow Editor", "Inspector"])

# 🔐 Initialize flow only once
if "curr_state" not in st.session_state:
    st.session_state.curr_state = StreamlitFlowState(
        nodes=[
            StreamlitFlowNode("1", (0, 0), {'content': 'Node 1'}, 'input', 'right'),
            StreamlitFlowNode("2", (1, 0), {'content': 'Node 2'}, 'default', 'right', 'left'),
        ],
        edges=[
            StreamlitFlowEdge("1-2", "1", "2", animated=True)
        ]
    )

if page == "Flow Editor":
    st.title("🧠 Flow Editor")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Node"):
            new_node = StreamlitFlowNode(
                str(uuid4()), (random.randint(0, 4), random.randint(0, 4)),
                {'content': f'Node {len(st.session_state.curr_state.nodes)+1}'},
                'default', 'right', 'left'
            )
            st.session_state.curr_state.nodes.append(new_node)
            st.rerun()

    with col2:
        if st.button("Reset Flow"):
            del st.session_state.curr_state
            st.rerun()

    # Render the flow diagram
    st.session_state.curr_state = streamlit_flow(
        "flow",
        st.session_state.curr_state,
        layout=ManualLayout(),  # Prevents layout from auto-resetting
        fit_view=True,
        height=500,
        allow_new_edges=True,
        get_node_on_click=True,
        get_edge_on_click=True,
        enable_node_menu=True,
        enable_edge_menu=True,
        show_minimap=True
    )

elif page == "Inspector":
    st.title("🛠️ Flow Inspector")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nodes")
        for node in st.session_state.curr_state.nodes:
            st.json(vars(node))

    with col2:
        st.subheader("Edges")
        for edge in st.session_state.curr_state.edges:
            st.json(vars(edge))

    st.subheader("Selected Element")
    st.write(st.session_state.curr_state.selected_id)
