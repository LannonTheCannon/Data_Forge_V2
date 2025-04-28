import streamlit as st
from streamlit_elements import elements, dashboard, html

# This callback updates our session state and logs changes in the browser.
def handle_layout_change(new_layout):
    st.session_state["layout"] = new_layout
    print("Layout changed:", new_layout)  # Shows updated coords in the Streamlit UI

if "layout" not in st.session_state:
    # We define one item with isDraggable=True & isResizable=True so the user can move it.
    st.session_state["layout"] = [
        dashboard.Item("box1", x=0, y=0, w=3, h=3, isDraggable=True, isResizable=True),
        dashboard.Item("box2", x=1, y=1, w=3, h=3, isDraggable=True, isResizable=True),
        dashboard.Item("box3", x=1, y=0, w=3, h=3, isDraggable=True, isResizable=True)
    ]

st.title("Minimal streamlit-elements Demo")

with elements("demo"):
    # onLayoutChange is triggered after you drop the item in a new position.
    with dashboard.Grid(
        layout=st.session_state["layout"],
        onLayoutChange=handle_layout_change,
        key="my_grid",
        # Optionally, you can set rowHeight, breakpoints, etc.
        # rowHeight=30
    ):
        # A simple gray box
        with html.Div(
            key="box1",
            style={
                "backgroundColor": "#DDD",
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
            },
        ):
            html.H4("Drag me around!")

        with html.Div(
            key="box2",
            style={
                "backgroundColor": "#DDD",
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
            },
        ):
            html.H4("Drag me around!")

            # A simple gray box
        with html.Div(
            key="box3",
            style={
                "backgroundColor": "#DDD",
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
            },
        ):
            html.H4("Drag me around!")
