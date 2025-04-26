import pandas as pd
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import openai
import numpy as np

# Node template (your custom classes)
from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode

# Agents
from ai_data_science_team.agents import DataCleaningAgent, FeatureEngineeringAgent

# LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini')

# OpenAI Client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Config
st.set_page_config(page_title="DataForge: Clean, Engineer, Map", layout="wide")

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

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts",
            "DATA_RAW", "plots", "dataframes", "msg_index", "clicked_questions", "dataset_name"]:
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
        response = openai.chat.completions.create(
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

# -------------- Page Layouts -------------- #

PAGE_OPTIONS = ['Data Upload', 'Mind Mapping']
page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

# -------------- Main -------------- #

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('🧹 DataForge Upload + Transformation')

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            try:
                # Step 1: Load dataset
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")

                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name

                # Step 2: Run Data Cleaning Agent
                with st.spinner('Cleaning Data...'):
                    data_cleaning_agent.invoke_agent(data_raw=df, user_instructions="Use default cleaning steps.")
                    df_cleaned = data_cleaning_agent.get_data_cleaned()

                # Step 3: Run Feature Engineering Agent
                with st.spinner('Engineering Features...'):
                    feature_engineering_agent.invoke_agent(data_raw=df_cleaned, user_instructions="Use default feature engineering steps.")
                    df_final = feature_engineering_agent.get_data_engineered()

                # Step 4: Update Session State
                st.session_state.df = df_final
                st.session_state.DATA_RAW = df_final
                st.session_state.df_preview = df_final.head()

                numeric_summary = df_final.describe()
                categorical_summary = df_final.describe(include=['object', 'category', 'bool'])

                numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_final.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                cat_cardinalities = {col: int(df_final[col].nunique()) for col in categorical_cols}
                top_cats = {col: df_final[col].value_counts().head(3).to_dict() for col in categorical_cols}

                st.session_state.df_summary = numeric_summary
                st.session_state.metadata_string = (
                    f"Columns: {list(df_final.columns)}\n"
                    f"Numeric columns: {numeric_cols}\n"
                    f"Categorical columns: {categorical_cols} (cardinalities: {cat_cardinalities})\n"
                    f"Top categories: {top_cats}\n"
                    f"Row count: {len(df_final)}"
                )

                # Step 5: Update Root Node
                root_question = generate_root_summary_question(st.session_state.metadata_string)
                if st.session_state.curr_state.nodes:
                    root_node = st.session_state.curr_state.nodes[0]
                    root_node.data["full_question"] = root_question
                    root_node.data["content"] = dataset_name
                    root_node.data["short_label"] = "ROOT"

                # Step 6: Show in Tabs
                tabs = st.tabs(["Raw Data", "Cleaned Data", "Feature Engineered Data", "Cleaning Agent Code", "Feature Engineering Code"])

                with tabs[0]:
                    st.subheader("STEP 1) Raw Uploaded Data Preview")
                    st.dataframe(df.head())

                with tabs[1]:
                    st.subheader("STEP 2) Cleaned Data Preview")
                    st.dataframe(df_cleaned.head())

                with tabs[2]:
                    st.subheader("STEP 3) Final Feature Engineered Data")
                    st.dataframe(df_final.head())
                    csv = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Cleaned Dataset", data=csv, file_name="dataforge_cleaned_dataset.csv", mime="text/csv")

                with tabs[3]:
                    st.subheader("STEP 4) Data Cleaning Agent - Generated Code")
                    cleaning_code = data_cleaning_agent.get_data_cleaner_function()
                    st.code(cleaning_code, language='python')

                with tabs[4]:
                    st.subheader("STEP 5) Feature Engineering Agent - Generated Code")
                    feature_code = feature_engineering_agent.get_feature_engineer_function()
                    st.code(feature_code, language='python')

            except Exception as e:
                st.error(f"Something went wrong: {e}")

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

        if st.session_state.clicked_questions:
            st.write("## Clicked Nodes (User's Exploration Path)")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            df_log = df_log[["section", "short_label", "node_type", "full_question"]]
            st.table(df_log)