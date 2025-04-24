import openai
import streamlit as st
import sqlalchemy as sql
from sqlalchemy.pool import NullPool
import pandas as pd
import asyncio
import os

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.agents import SQLDatabaseAgent

# * APP Inputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "northwind.db"))
DB_OPTIONS = {"Northwind Database": f"sqlite:///{DB_PATH}"}

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']
TITLE = "Your SQL Database Agent"

# Streamlit setup
st.set_page_config(page_title=TITLE, page_icon="📊")
st.title(TITLE)
st.markdown(
    """
    Welcome to the SQL Database Agent. This AI agent is designed to help you query your SQL database
    and return data frames that you can interactively inspect and download.
    """
)

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tables exist in the database?
        - What are the first 10 rows in the territory table?
        - Aggregate sales for each territory.
        - Aggregate sales by month for each territory.
        """
    )

# Sidebar: Database selection and engine creation
db_option = st.sidebar.selectbox("Select a Database", list(DB_OPTIONS.keys()))
st.session_state["PATH_DB"] = DB_OPTIONS[db_option]
st.write(f"Resolved DB path: `{DB_PATH}`")
st.write("File exists?", os.path.exists(DB_PATH))

sql_engine = sql.create_engine(
    DB_OPTIONS[db_option],
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)

# OpenAI setup
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)
OPENAI_LLM = ChatOpenAI(model=model_option, api_key=st.secrets["OPENAI_API_KEY"])
llm = OPENAI_LLM

# Chat memory and dataframe store
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if not msgs.messages:
    msgs.add_ai_message("How can I help you?")

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Function to render chat + stored dataframes
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "DATAFRAME_INDEX:" in msg.content:
                idx = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[idx])
            else:
                st.write(msg.content)

display_chat_history()

# Async handler using fresh connection per query
def run_agent(question: str):
    async def _inner():
        with sql_engine.connect() as conn:
            agent = SQLDatabaseAgent(
                model=llm,
                connection=conn,
                n_samples=1,
                log=False,
                bypass_recommended_steps=True,
            )
            await agent.ainvoke_agent(user_instructions=question)
            return agent
    return asyncio.run(_inner())

# Main user input loop
if st.session_state["PATH_DB"] and (question := st.chat_input("Enter your question here:", key="query_input")):
    if not st.secrets["OPENAI_API_KEY"]:
        st.error("Please set your OpenAI API key in secrets.")
        st.stop()

    st.chat_message("human").write(question)
    msgs.add_user_message(question)

    with st.spinner("Thinking..."):
        try:
            result = run_agent(question)
            # Safely extract SQL and DataFrame
            sql_query = result.get_sql_query_code()
            if sql_query is None:
                sql_query = ""
            tmp_df = result.get_data_sql()
            response_df = tmp_df if tmp_df is not None else pd.DataFrame()
        except Exception as e:
            error_msg = f"I'm sorry, I couldn't process that. Error: {e}"
            msgs.add_ai_message(error_msg)
            st.chat_message("ai").write(error_msg)
            st.error(error_msg)
            raise

    # Record and display result
    header = f"### SQL Results:\n\nSQL Query:\n```sql\n{sql_query}\n```\n\nResult:"
    df_index = len(st.session_state.dataframes)
    st.session_state.dataframes.append(response_df)
    msgs.add_ai_message(header)
    msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
    st.chat_message("ai").write(header)

    if not response_df.empty:
        st.dataframe(response_df)
    else:
        st.warning("No data returned. Check the query or the database content.")