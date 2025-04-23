from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode
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


st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# ##################### SESSION STATE STUFF ####################### #

if "curr_state" not in st.session_state:
    # Prepare root node. We'll store the dataset metadata in "full_question" if we have it.
    dataset_label = st.session_state.get("dataset_name", "Dataset")

    root_theme = ThemeNode(
        node_id="S0",
        label="ROOT",
        full_question="Overview of the dataset",
        category="Meta",
        node_type="theme",
        parent_id=None,
        metadata={"content": dataset_label}
    )

    st.session_state.mindmap_nodes = {"S0": root_theme}
    st.session_state.curr_state = StreamlitFlowState(
        nodes=[root_theme.to_streamlit_node()],
        edges=[]
    )

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index", "clicked_questions", "dataset_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "DATA_RAW"] else []

if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()

if "seen_embeddings" not in st.session_state:
    st.session_state.seen_embeddings = []

# ######################## UPLOAD DATA ######################## #

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
            model="gpt-4.1-mini",
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

# ######################## DATA ANALYST ######################## #

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
                        tabs = st.tabs(["📊 Output", "📋 Data Preview", "💻 Code"])

                        # --- Code Tab First, to capture edits and trigger updates ---
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
                        with tabs[1]:
                            df_preview = artifact.get("data_preview")
                            if df_preview is not None:
                                st.write("### Data‑Wrangler Output")
                                st.dataframe(df_preview, use_container_width=True)
                            else:
                                st.write("_No data preview available._")

                        with tabs[2]:
                            code_before = st.session_state.get(editor_key, artifact.get("code", ""))
                            editor_response = code_editor(
                                code=code_before,
                                lang="python",
                                theme="dracula",
                                height=300,
                                # buttons=[
                                #     {
                                #         "name": "Run",
                                #         "feather": "Play",
                                #         "primary": True,
                                #         "hasText": True,
                                #         "showWithIcon": True,
                                #         "commands": ["submit"],
                                #         "style": {"bottom": "0.44rem", "right": "0.4rem"}
                                #     }
                                # ],
                                key=f"code_editor_{unique_key}"
                            )

                            new_code = editor_response.get("text", "").strip()


                            #
                            #
                            # # Only run if the code has changed
                            # if new_code and new_code != st.session_state.get(editor_key):
                            #     try:
                            #         exec_globals = {
                            #             "df": st.session_state.df,
                            #             "pd": pd,
                            #             "np": np,
                            #             "sns": sns,
                            #             "go": go,
                            #             "plt": plt,
                            #             "pio": pio,
                            #             "st": st,
                            #             "json": json
                            #         }
                            #         exec_locals = {}
                            #         exec(new_code, exec_globals, exec_locals)
                            #
                            #         output_obj = exec_locals.get("fig") or \
                            #                      exec_locals.get("output") or \
                            #                      exec_locals.get("fig_dict")
                            #
                            #         if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                            #             output_obj = pio.from_json(json.dumps(output_obj))
                            #
                            #         artifact["data"] = output_obj
                            #         artifact["render_type"] = "plotly" if isinstance(output_obj, go.Figure) else "dataframe"
                            #         st.session_state[editor_key] = new_code
                            #         st.session_state[output_key] = output_obj
                            #
                            #     except Exception as e:
                            #         st.error(f"Error executing code: {e}")


PAGE_OPTIONS = [
    'Data Upload',
    'Mind Mapping',
    'Data Analyst'
]

page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('Upload your own Dataset!')

        upload_type = st.radio('Upload Method', ['Upload File', 'Paste S3 URL', 'Connect DB'])

        if upload_type == 'Upload File':
            uploaded_file = st.file_uploader('Choose a CSV', type=['csv', 'excel'])
            if uploaded_file is not None:
                # Load data into session state
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state["DATA_RAW"] = df
                    st.session_state.df_preview = df.head()
                    # st.session_state.df_summary = df.describe()

                    # Save dataset name without extension
                    dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                    st.session_state['dataset_name'] = dataset_name
                    # st.write(dataset_name)


                    # numeric + categorical summary
                    numeric_summary = df.describe()
                    cat_summary = df.describe(include=['object', 'category', 'bool'])
                    # build a richer metadata string
                    st.session_state.df_summary = numeric_summary  # keep for display
                    cols = df.columns.tolist()
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                    cat_cardinalities = {c: int(df[c].nunique()) for c in categorical_cols}
                    top_cats = {c: df[c].value_counts().head(3).to_dict() for c in categorical_cols}

                    # Rebuild a 'metadata_string' for the root node
                    if st.session_state.df_summary is not None:
                        # Basic example of turning summary + columns into a string
                        cols = list(st.session_state.df_summary.columns)
                        row_count = st.session_state.df.shape[0]
                        st.session_state.metadata_string = (
                            f"Columns: {cols}\n"
                            f"Numeric columns: {numeric_cols}\n"
                            f"Categorical columns: {categorical_cols} (cardinalities: {cat_cardinalities})\n"
                            f"Top categories: {top_cats}\n"
                            f"Row count: {len(df)}"
                        )
                        # print(st.session_state.metadata_string)
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

        elif upload_type == 'Paste S3 URL':
            s3_url = st.text_input("S3 Parquet file URL")
        elif upload_type == "Connect DB":
            db_uri = st.text_input("Enter your DB connection string")

        # uploaded_file = st.file_uploader('Upload CSV or Excel here', type=['csv', 'excel'])

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

        # ------------------ Mind Map Node Click Event Handler ------------------

        clicked_node_id = st.session_state.curr_state.selected_id

        if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
            st.write("👉 Node clicked:", clicked_node_id)

            # Get the corresponding object from our mindmap node registry
            clicked_obj = st.session_state.mindmap_nodes.get(clicked_node_id)

            # Log for debugging
            st.write("📌 All registered node IDs:", list(st.session_state.mindmap_nodes.keys()))
            st.write("📌 Already expanded nodes:", st.session_state.expanded_nodes)

            if not clicked_obj:
                st.warning(f"⚠️ Node '{clicked_node_id}' not found in mindmap_nodes.")
            else:
                # ✅ Only log if not already recorded
                already_logged = any(q["section"] == clicked_obj.node_id for q in st.session_state.clicked_questions)

                if not already_logged:
                    st.session_state.clicked_questions.append({
                        "section": clicked_obj.node_id,
                        "short_label": clicked_obj.label,
                        "full_question": clicked_obj.full_question,
                        "node_type": clicked_obj.node_type
                    })

                if clicked_obj.can_expand() and clicked_node_id not in st.session_state.expanded_nodes:
                    children = clicked_obj.get_children(
                        openai_client=client,
                        metadata_string=st.session_state.metadata_string
                    ) or []

                    for child in children:
                        st.session_state.mindmap_nodes[child.node_id] = child
                        st.session_state.curr_state.nodes.append(child.to_streamlit_node())
                        st.session_state.curr_state.edges.append(
                            StreamlitFlowEdge(
                                f"{clicked_obj.node_id}-{child.node_id}",
                                clicked_obj.node_id,
                                child.node_id,
                                animated=True
                            )
                        )

                    clicked_obj.mark_expanded()
                    st.session_state.expanded_nodes.add(clicked_node_id)

                st.rerun()

        # ------------------ Display User Click History ------------------

        if st.session_state.clicked_questions:
            st.write("## Clicked Nodes (User's Exploration Path)")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            df_log = df_log[["section", "short_label", "node_type", "full_question"]]
            st.table(df_log)

    elif page == 'Data Analyst':
        st.subheader('Pandas Data Analyst Mode')
        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
        if len(msgs.messages) == 0:
            pass
            # msgs.add_ai_message("IMPORTANT: For best results use this formula -> Create a [chart] of the [field] on the y-axis (aggregation) and the [field] on the x-axis and make the chart [color].")
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4.1-mini', api_key=st.secrets['OPENAI_API_KEY'])
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
            msgs.add_user_message(question)
            with st.spinner("Thinking..."):
                try:
                    st.session_state.pandas_data_analyst.invoke_agent(
                        user_instructions=question,
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
                        df_wrangled  = result.get('data_wrangled')

                        # Combine both functions into one code block
                        combined_code = f"{wrangle_code}\n\n{viz_code}\n\n# Runtime Execution\noutput = data_visualization(data_wrangler([df]))"

                        st.session_state["chat_artifacts"][msg_index].append({
                            "title": "Chart",
                            "render_type": "plotly",
                            "data": plot_obj,
                            "code": combined_code,
                            "data_preview": df_wrangled
                        })
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # print(result['data_visualization_function'])

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