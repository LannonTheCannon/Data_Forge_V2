import streamlit as st

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import RadialLayout



nodes = [StreamlitFlowNode("main",
                           (0, 0),
                           {'content':"# Markdown Support in Nodes"},
                           'input',
                           'bottom',
                           style={'backgroundColor': 'red'}),

        StreamlitFlowNode(
                    'hello',
                    (1, 0),
                    {"content": '# Markdown world'},
                    "input", "right", "left",
                    style={"backgroundColor": 'green'}
                ),


        StreamlitFlowNode("text",
                          (0, 0),
                          {'content':
    """### Text
Can support markdown text styles: **bold**, *italic* and `code`"""},
                          'output',
                          'bottom',
                          'top'),


        StreamlitFlowNode("code",
                          (0, 0),
                          {'content':
"""### Code Block 
```python
print('Hello World')
```"""},
                          'output',
                          'bottom',
                          'top'),

        StreamlitFlowNode("list",
                          (0, 0),
                          {'content':
"""### List
1. Ordered
2. And
- Unordered
- Lists
"""},
                          'output',
                          'bottom',
                          'top'),

        StreamlitFlowNode("math",
                          (0, 0),
                          {'content':
"""### Math
$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$
"""},
                          'output',
                          'bottom',
                          'top'),

        StreamlitFlowNode("github",
                          (0, 0),
                          {'content': '## Github Flavor Support'},
                          'default',
                          'top',
                          'bottom'),

        StreamlitFlowNode("strikethrough",
                          (0, 0),
                          {'content':
"""
### ~Strike through~
"""},
                          'output',
                          'bottom',
                          'top'),

        StreamlitFlowNode("table",
                          (0, 0),
                          {'content':
"""### Table

| a | b  |  c |  d  |
| - | :- | -: | :-: |
| 1 | 2 | 3 | 4 |
| 5 | 6 | 7 | 8 |

"""},
                          'output',
                          'top',
                          'right'),

        StreamlitFlowNode("tasks",
                          (0, 0),
                          {'content':
"""## Tasklist

* [ ] to do
* [x] done
"""},
                          'output',
                          'top',
                          'bottom'),

        StreamlitFlowNode("html",
                          (0, 0),
                          {'content':
"""## Raw HTML"""},
                          'default',
                          'top',
                          'bottom'),

        StreamlitFlowNode("link",
                          (0, 0),
                          {'content':
"""### Link
<a href="https://github.com/dkapur17/streamlit-flow" target="_blank">Streamlit Flow</a>"""},
                          'output',
                          'top',
                          'bottom'),

        StreamlitFlowNode("expander",
                          (0, 0),
                          {'content':
"""### Expander
<details>
<summary>Click to expand</summary>

This is hidden content
</details>
"""},
                          'output',
                          'top',
                          'bottom'),

        StreamlitFlowNode("image",
                          (0, 0),
                          {'content':
"""### Image
<img src="https://raw.githubusercontent.com/dkapur17/streamlit-flow/master/assets/streamlit-flow-logo.svg" alt="Streamlit Flow Logo" width="100">
"""},
                          'output',
                          'top',
                          'bottom'),

        StreamlitFlowNode("video",
                          (0, 0),
                          {'content':
"""### Video
<video width="256" controls>
    <source src="https://github.com/dkapur17/streamlit-flow/raw/master/assets/FastBuild.mp4" type="video/mp4">
</video>"""},
                          'output',
                          'bottom',
                          'top')]

edges = [StreamlitFlowEdge("main-text", "main", "text", animated=True),
        StreamlitFlowEdge("main-code", "main", "code", animated=True),
        StreamlitFlowEdge("main-list", "main", "list", animated=True),
        StreamlitFlowEdge("main-math", "main", "math", animated=True),
        StreamlitFlowEdge("main-github", "main", "github", animated=True),
        StreamlitFlowEdge("github-strikethrough", "github", "strikethrough", animated=True),
        StreamlitFlowEdge("github-table", "github", "table", animated=True),
        StreamlitFlowEdge("github-tasks", "github", "tasks", animated=True),
        StreamlitFlowEdge("main-html", "main", "html", animated=True),
        StreamlitFlowEdge("html-link", "html", "link", animated=True),
        StreamlitFlowEdge("html-expander", "html", "expander", animated=True),
        StreamlitFlowEdge("html-image", "html", "image", animated=True),
        StreamlitFlowEdge("html-video", "html", "video", animated=True)]

if 'markdown_node_state' not in st.session_state:
    st.session_state.markdown_node_state = StreamlitFlowState(nodes, edges)

streamlit_flow('markdown_node_flow',
               st.session_state.markdown_node_state,
               layout=RadialLayout(),
               fit_view=True,
               height=1000,
               show_controls=True,
               show_minimap=True,
               allow_new_edges=True,
               animate_new_edges=True,
               get_node_on_click=True,
               get_edge_on_click=True,
               pan_on_drag=True,
               allow_zoom=True,
               min_zoom=0.5,
               enable_pane_menu=True,
               enable_node_menu=True,
               enable_edge_menu=True,
)
#
# streamlit_flow(
#     key:str, #
#     state:StreamlitFlowState, #
#     height:int=500, #
#     fit_view:bool=False, #
#     show_controls:bool=True,
#     show_minimap:bool=False,
#     allow_new_edges:bool=False,
#     animate_new_edges:bool=False,
#     style:dict={},
#     layout:Layout=ManualLayout(),
#     get_node_on_click:bool=False,
#     get_edge_on_click:bool=False,
#     pan_on_drag:bool=True,
#     allow_zoom:bool=True,
#     min_zoom:float=0.5,
#     enable_pane_menu:bool=False,
#     enable_node_menu:bool=False,
#     enable_edge_menu:bool=False,
#     hide_watermark:bool=False
# )