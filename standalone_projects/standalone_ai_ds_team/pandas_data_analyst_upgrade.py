from openai import OpenAI
import openai
import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import plotly.express as px

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict


def process_routed_analysis(question, llm, data, analysis_type="Auto-detect"):
    """
    Unified exploratory analysis handler that:
    - maintains artifact-based architecture from EDAToolsAgent
    - Adds routing support for tables and charts like in pandasDataAnalyst
    """
    eda_agent = EDAToolsAgent(
        llm,
        invoke_react_agent_kwargs={'recursion_limit': 10},
    )

    # Append routing hints to question
    if analysis_type == 'Chart':
        question += ' Return this as a chart if possible.'
    elif analysis_type == 'Table':
        question += ' Return this as a data table if possible'
    else:
        question += ' Use the most appropriate format (Chart or table).'

    question += "Don't return hyperlinks to files in this response"

    eda_agent.invoke_agent(user_instructions=question, data_raw=data)

    tool_calls = eda_agent.get_tool_calls()
    ai_message = eda_agent.get_ai_message(markdown=False)
    artifacts = eda_agent.get_artifacts(as_dataframe=False)

    result = {
        'ai_message': ai_message,
        'tool_calls': tool_calls,
        'artifacts': artifacts
    }

    return result


def build_pandas_data_analyst(llm, log=False):
    """Convenience initializer to mirror the standalone_ai_ds_team build."""
    return PandasDataAnalyst(
        model=llm,
        data_wrangling_agent=DataWranglingAgent(
            model=llm,
            log=log,
            bypass_recommended_steps=True,
            n_samples=100,
        ),
        data_visualization_agent=DataVisualizationAgent(
            model=llm,
            n_samples=100,
            log=log,
        ),
    )


# APP Inputs
MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']
TITLE = "Pandas Data Analyst AI Copilot"

# Streamlit APP Config
st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)

st.title(TITLE)
st.markdown('''
Pandas Data Analyst AI. Upload a CSV or EXCEL file and ask questions about the data. 
The AI Agent will analyze your dataset and return either data tables or interactive charts.
''')

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        ##### Bikes Data Set: 

        - Show the top 5 bike models by extended sales. 
        - Show the top 5 bike models by extended sales in a bar chart. 
        - Show the top 5 bike models extended sales in a pie chart. 
        - Make a plot of extended sales by month for each bike model. Use a color to identify the bike models. 
        """
    )

# Choose OpenAI Model
model_option = st.sidebar.selectbox("Choose OpenAI Model", MODEL_LIST, index=0)

# OpenAI API Key entry and test
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
llm = ChatOpenAI(model=model_option, api_key=st.secrets['OPENAI_API_KEY'])

# File Upload and Data Preview
st.markdown("Upload a CSV or Excel file and ask questions about your data.")
uploaded_file = st.file_uploader('Choose a CSV or EXCEL', type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader('Data Preview')
    st.dataframe(df.head())
else:
    st.info('Please upload a CSV or Excel file to get started!')
    st.stop()

# Initialize Chat Message History and Storage
msgs = StreamlitChatMessageHistory(key='langchain_messages')
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

if 'plots' not in st.session_state:
    st.session_state.plots = []

if 'dataframes' not in st.session_state:
    st.session_state.dataframes = []


def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                )
            else:
                st.write(msg.content)


# Render current messages from StreamlitChatMessageHistory
display_chat_history()

# ---------------------------
# AI Agent Setup
# ---------------------------
LOG = False

pandas_data_analyst = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_visualization_agent=DataVisualizationAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
)

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------
analysis_type = st.selectbox('Preferred Output Format', ['Auto-detect', 'Chart', 'Table'])

if question := st.chat_input("Enter your question here:", key="query_input"):
    if not st.secrets["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()

    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        try:
            # Invoke agent for routing analysis
            result = process_routed_analysis(question, llm, df, analysis_type)

        except Exception as e:
            st.chat_message("ai").write("An error occurred while processing your query. Please try again.")
            msgs.add_ai_message("An error occurred while processing your query. Please try again.")
            st.stop()

        routing = result.get("routing_preprocessor_decision")

        if routing == "chart" and not result.get("plotly_error", False):
            # Process chart result
            plot_data = result.get("plotly_fig")
            if plot_data:
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                response_text = "Returning the generated chart."
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                st.chat_message("ai").write(response_text)
                st.plotly_chart(plot_obj)
            else:
                st.chat_message("ai").write("The agent did not return a valid chart.")
                msgs.add_ai_message("The agent did not return a valid chart.")

        elif routing == "table":
            # Process table result
            data_wrangled = result.get("dataframe")
            if data_wrangled is not None:
                response_text = "Returning the data table."
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                st.chat_message("ai").write("No table data was returned by the agent.")
                msgs.add_ai_message("No table data was returned by the agent.")
        else:
            # Fallback if routing decision is unclear or if chart error occurred
            data_wrangled = result.get("dataframe")
            if data_wrangled is not None:
                response_text = "I apologize. There was an issue with generating the chart. Returning the data table instead."
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                response_text = "An error occurred while processing your query. Please try again."
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)

        # Manually calculate and plot average salary by experience level
        if "salary_in_usd" in df.columns and "experience_level" in df.columns:
            salary_by_experience = df.groupby('experience_level')['salary_in_usd'].mean().reset_index()
            fig = px.bar(salary_by_experience, x='experience_level', y='salary_in_usd',
                         title="Average Salary by Experience Level")
            st.plotly_chart(fig)
