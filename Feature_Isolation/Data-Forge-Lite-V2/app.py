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

# Node template (your custom classes)
from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode

# Agents
from ai_data_science_team.agents import DataCleaningAgent, FeatureEngineeringAgent

# LLM
from langchain_openai import ChatOpenAI
    # Create the ChatOpenAI instance NOW



st.sidebar.header('Enter your OpenAI API Key')

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = None

st.session_state['OPENAI_API_KEY'] = st.sidebar.text_input(
    "API KEY",
    type="password",
    help="Your OpenAI API key is required for the app to function."
)

# Test OpenAI API Key
if st.session_state['OPENAI_API_KEY']:
    # Set the API key for OPENAI


    # Test the API key (optional)
    try:
        llm = ChatOpenAI(
        model='gpt-4o-mini',
        openai_api_key=st.session_state['OPENAI_API_KEY']
        )
        client = openai.OpenAI(
        api_key=st.session_state['OPENAI_API_KEY']
)
        # Example: Fetch models to validate the key
        # models = client.models.list()
        # st.success("API Key is valid!")

    except Exception as e:
        st.error(f"Invalid API key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed!")
    st.stop()

# OpenAI Client
# client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Config
# st.set_page_config(page_title="DataForge: Clean, Engineer, Map", layout="wide")

# Colors for mindmap
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# Initialize Agents
data_cleaning_agent = DataCleaningAgent(model=llm, n_samples=50, log=False)
feature_engineering_agent = FeatureEngineeringAgent(model=llm, n_samples=50, log=False)

# -------------- Session State Initialization -------------- #

if "curr_state" not in st.session_state:
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
    st.session_state.curr_state = StreamlitFlowState(nodes=[root_theme.to_streamlit_node()], edges=[])

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index", "clicked_questions", "dataset_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "DATA_RAW"] else []

if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()

if "seen_embeddings" not in st.session_state:
    st.session_state.seen_embeddings = []

# -------------- Utility Functions -------------- #

def generate_root_summary_question(metadata_string: str) -> str:
    if not metadata_string:
        return "Overview of the dataset"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a data summarizer."},
                {"role": "user", "content": f"Dataset metadata:\n{metadata_string}\n\nSummarize in one sentence."}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip().split("\n")[0]
    except Exception:
        return "Overview of the dataset"

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
        response = client.chat.completions.create(
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
                            code_editor(
                                code=code_before,
                                lang="python",
                                theme="dracula",
                                height=300,
                                key=f"code_editor_{unique_key}"
                            )

                        # 🔥 New: Add to Workshop Button
                        if st.button("➕ Add to Workshop", key=f"add_to_workshop_{i}_{j}"):
                            st.session_state.workshop_item = {
                                "chart": artifact.get("data"),
                                "data_preview": artifact.get("data_preview"),
                                "code": artifact.get("code")
                            }
                            st.success("✅ Added to Workspace! Go to the Workspace tab.")

def workshop_page():
    import json
    import plotly.io as pio
    from code_editor import code_editor
    from langchain_openai import ChatOpenAI

    st.title("🔧 DataForge Workshop")

    # 1) Nothing to do if no workshop item
    if not st.session_state.get("workshop_item"):
        st.info(
            "Nothing in the Workshop yet. Go to the Data Analyst Agent or Mind Mapping page "
            "and click ‘➕ Add to Workshop’ on an artifact."
        )
        return

    item = st.session_state.workshop_item

    # 2) Initialize the code buffer on first load
    if "workshop_code_text" not in st.session_state:
        st.session_state.workshop_code_text = item.get("code", "")

    # 3) Create four tabs
    chart_tab, table_tab, code_tab, ai_tab = st.tabs(
        ["📈 Chart", "📋 Table", "💻 Code", "✍️ AI Request"]
    )

    # 4) Placeholders for chart & table
    with chart_tab:
        chart_ph = st.empty()
    with table_tab:
        table_ph = st.empty()

    # 5) Initial render of existing chart + table
    fig = item.get("chart")
    if fig:
        if isinstance(fig, dict) and "data" in fig and "layout" in fig:
            fig = pio.from_json(json.dumps(fig))
        chart_ph.plotly_chart(fig, use_container_width=True)
    else:
        chart_ph.info("No chart available yet.")

    df_prev = item.get("data_preview")
    if df_prev is not None:
        table_ph.dataframe(df_prev, use_container_width=True)
    else:
        table_ph.info("No data preview available.")

    # 6) Code tab: embedded Run button in the editor
    with code_tab:
        editor_response = code_editor(
            code=st.session_state.workshop_code_text,
            lang="python",
            theme="dracula",
            height=400,
            key="workshop_code_editor_block",
            buttons=[{
                "name":    "▶️ Run",    # label
                "feather": "Play",      # icon
                "primary": True,
                "commands": ["submit"],
                "style":   {"bottom": "0.5rem", "right": "0.5rem"}
            }],
        )

        new_code = editor_response.get("text", "")

        # Only re-run if the user actually clicked Run (text changed)
        if new_code and new_code != st.session_state.workshop_code_text:
            st.session_state.workshop_code_text = new_code

            try:
                # Execute the updated code
                exec_globals = {
                    "df": st.session_state.get("df"),
                    "pd": pd, "np": np, "sns": sns,
                    "go": go, "plt": plt, "pio": pio,
                    "st": st, "json": json,
                }
                exec_locals = {}
                exec(new_code, exec_globals, exec_locals)

                # Capture new figure if any
                out = (
                    exec_locals.get("fig")
                    or exec_locals.get("output")
                    or exec_locals.get("fig_dict")
                )
                if out:
                    if isinstance(out, dict) and "data" in out and "layout" in out:
                        out = pio.from_json(json.dumps(out))
                    item["chart"] = out
                    chart_ph.empty()
                    chart_ph.plotly_chart(out, use_container_width=True)

                # Capture new dataframe if any
                new_df = exec_locals.get("df")
                if new_df is not None:
                    item["data_preview"] = new_df
                    table_ph.empty()
                    table_ph.dataframe(new_df, use_container_width=True)

                # Persist changes
                item["code"] = new_code
                st.session_state.workshop_item = item

                st.success("✅ Code executed—chart & table updated!")
            except Exception as e:
                st.error(f"Error running updated code: {e}")

    # 7) AI-assisted rewrite tab
    with ai_tab:
        st.write("### ✍️ Request AI to update your code")
        user_req = st.text_input("Enter your modification request:", key="ai_request")
        if st.button("🤖 Rewrite Code with AI"):
            if not user_req:
                st.warning("Please enter a request first!")
            else:
                try:
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        openai_api_key=st.session_state["OPENAI_API_KEY"],
                    )
                    resp = llm.invoke(f"""
Please take the following Python code and modify it based on this request:
Request: {user_req}

Python Code:
{item['code']}

Provide only the modified code, without extra explanations.
""")
                    rewritten = resp.content.strip()

                    # Persist AI rewrite and reload editor buffer
                    item["code"] = rewritten
                    st.session_state.workshop_code_text = rewritten
                    st.session_state.workshop_item = item

                    st.success("✅ Code updated by AI! Switch to the Code tab to review.")
                except Exception as e:
                    st.error(f"Error updating code with AI: {e}")

PAGE_OPTIONS = ['Data Upload', 'Mind Mapping', "Data Analyst Agent", "Workspace"]
page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

# -------------- Main -------------- #

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('🧹 DataForge Upload + Transformation')

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # If a new file uploaded → process it
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # st.success("✅ File uploaded successfully!")

                dataset_name = uploaded_file.name.rsplit('.', 1)[0]

                # Cleaning
                with st.spinner('🧹 Cleaning Data...'):
                    data_cleaning_agent.invoke_agent(
                        data_raw=df,
                        user_instructions="Use default cleaning steps."
                    )
                    df_cleaned = data_cleaning_agent.get_data_cleaned()

                # Feature Engineering
                with st.spinner('🛠️ Engineering Features...'):
                    feature_engineering_agent.invoke_agent(
                        data_raw=df_cleaned,
                        user_instructions="Use default feature engineering steps."
                    )
                    df_final = feature_engineering_agent.get_data_engineered()

                # Save everything into session_state
                st.session_state.df = df_final
                st.session_state.DATA_RAW = df_final
                st.session_state.df_preview = df_final.head()

                st.session_state.df_uploaded_raw = df  # <-- new: raw uploaded file
                st.session_state.df_cleaned = df_cleaned
                st.session_state.df_final = df_final
                st.session_state.cleaning_code = data_cleaning_agent.get_data_cleaner_function()
                st.session_state.feature_engineering_code = feature_engineering_agent.get_feature_engineer_function()

                numeric_summary = df_final.describe()
                # categorical_summary = df_final.describe(include=['object', 'category', 'bool'])

                #############
                numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_final.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                # Protect against weird columns
                safe_categorical_cols = []
                cat_cardinalities = {}
                top_cats = {}

                for col in categorical_cols:
                    try:
                        nunique = int(df_final[col].nunique())
                        top_values = df_final[col].value_counts().head(3).to_dict()

                        cat_cardinalities[col] = nunique
                        top_cats[col] = top_values
                        safe_categorical_cols.append(col)

                    except Exception as e:
                        st.warning(f"Skipping column `{col}` due to error: {e}")

                # Now use only the safe columns
                st.session_state.df_summary = numeric_summary
                st.session_state.metadata_string = (
                    f"Columns: {list(df_final.columns)}\n"
                    f"Numeric columns: {numeric_cols}\n"
                    f"Categorical columns: {safe_categorical_cols} (cardinalities: {cat_cardinalities})\n"
                    f"Top categories: {top_cats}\n"
                    f"Row count: {len(df_final)}"
                )
                # st.write(st.session_state.metadata_string)

                root_question = generate_root_summary_question(st.session_state.metadata_string)
                if st.session_state.curr_state.nodes:
                    root_node = st.session_state.curr_state.nodes[0]
                    root_node.data["full_question"] = root_question
                    root_node.data["content"] = dataset_name
                    root_node.data["short_label"] = "ROOT"

            except Exception as e:
                st.error(f"Something went wrong during upload: {e}")

        # --- Now, if data already exists in session_state, show it ---
        if "df_uploaded_raw" in st.session_state:
            tabs = st.tabs([
                "📂 Raw Uploaded Data",
                "🧹 Cleaned Data",
                "🛠️ Feature Engineered Data",
                "📜 Cleaning Agent Code",
                "📜 Feature Engineering Code"
            ])

            with tabs[0]:
                st.subheader("STEP 1) Raw Uploaded Data Preview")
                st.dataframe(st.session_state.df_uploaded_raw.head())

            with tabs[1]:
                st.subheader("STEP 2) Cleaned Data Preview")
                st.dataframe(st.session_state.df_cleaned.head())

            with tabs[2]:
                st.subheader("STEP 3) Final Feature Engineered Data")
                st.dataframe(st.session_state.df_final.head())

                csv = st.session_state.df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=csv,
                    file_name="dataforge_cleaned_dataset.csv",
                    mime="text/csv"
                )

            with tabs[3]:
                st.subheader("STEP 4) Data Cleaning Agent - Generated Code")
                st.code(st.session_state.cleaning_code, language='python')

            with tabs[4]:
                st.subheader("STEP 5) Feature Engineering Agent - Generated Code")
                st.code(st.session_state.feature_engineering_code, language='python')

    elif page == 'Mind Mapping':
        st.title('🧠 Mind Mapping + Agentic Exploration')

        if st.session_state.get("dataset_name"):
            root_node = st.session_state.curr_state.nodes[0]
            if root_node.data["content"] != st.session_state["dataset_name"]:
                root_node.data["content"] = st.session_state["dataset_name"]

        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("🔄 Reset Mind Map"):
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

        # Render mind map
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

        # Node click event
        clicked_node_id = st.session_state.curr_state.selected_id

        if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
            clicked_obj = st.session_state.mindmap_nodes.get(clicked_node_id)

            if clicked_obj:
                # ✨ Log clicked node if not already recorded
                already_logged = any(q["section"] == clicked_obj.node_id for q in st.session_state.clicked_questions)
                if not already_logged:
                    st.session_state.clicked_questions.append({
                        "section": clicked_obj.node_id,
                        "short_label": clicked_obj.label,
                        "node_type": clicked_obj.node_type,
                        "full_question": clicked_obj.full_question
                    })

                # ✅ Expand children if needed
                if clicked_obj.can_expand() and clicked_node_id not in st.session_state.expanded_nodes:
                    children = clicked_obj.get_children(openai_client=client, metadata_string=st.session_state.metadata_string) or []

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
        # --- Always show the Clicked Questions Table, even if empty
        st.write("## 🧠 Your Exploration Path")

        if st.session_state.clicked_questions:
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            df_log = df_log[["section", "short_label", "node_type", "full_question"]]
            st.dataframe(df_log, use_container_width=True)
        else:
            st.info("Start clicking nodes on the mind map to populate your exploration path!")

    elif page == 'Data Analyst Agent':
        st.subheader('Pandas Data Analyst Mode')
        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
        if len(msgs.messages) == 0:
            pass
            # msgs.add_ai_message("IMPORTANT: For best results use this formula -> Create a [chart] of the [field] on the y-axis (aggregation) and the [field] on the x-axis and make the chart [color].")
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4.1-mini',
                               api_key=st.session_state['OPENAI_API_KEY'])
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

    elif page == "Workspace":
        workshop_page()