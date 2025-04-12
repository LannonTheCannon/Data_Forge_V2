from openai import OpenAI

import openai
import streamlit as st
import pandas as pd
import plotly.io as pio
import json
from IPython.display import Markdown

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI


from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)

# APP Inputs

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']
TITLE = "Pandas Data Analyst AI Copilot"

# Streamlit APP Config

st.set_page_config(
    page_title = TITLE,
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

model_option = st.sidebar.selectbox("Choose OpenAI Model", MODEL_LIST, index=0)
# OpenAI API Key entry and test
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
llm = ChatOpenAI(model=model_option, api_key=st.secrets['OPENAI_API_KEY'])

# File Upload and Data Preview
st.markdown(
    """
    Upload a CSV or Excel file and ask questions about your data. 
    The AI Agent will analyze your dataset and return either data tables or interactive charts
    """
)

uploaded_file = st.file_uploader(
    'Choose a CSV or EXCEL', type=['csv', 'xlsx', 'xls']
)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        pd.read_excel(uploaded_file)

    st.subheader('Data Preview')
    st.dataframe(df.head())

else:
    st.info('Please upload a CSV or Excel file to get started!')
    st.stop()

# Initialize Chat Message History and Storage
msgs = StreamlitChatMessageHistory(key='langchain_messages')
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can i help you?")

if 'plots' not in st.session_state:
    st.session_state.plots = []

if 'dataframes' not in st.session_state:
    st.session_state.dataframes = []

if "chat_artifacts" not in st.session_state:
    st.session_state['chat_artifacts'] = {}


def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            # Splitting the content and fetching the correct plotly chart
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                )

            # Getting the content and fetching the correct dataframe
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

if question := st.chat_input("Enter your question here:", key="query_input"):
    if not st.secrets["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()

    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        try:
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=df,
            )
            result = pandas_data_analyst.get_response()
        except Exception as e:
            st.chat_message("ai").write(
                "An error occurred while processing your query. Please try again."
            )
            msgs.add_ai_message(
                "An error occurred while processing your query. Please try again."
            )
            st.stop()

        routing = result.get("routing_preprocessor_decision")

        if routing == "chart" and not result.get("plotly_error", False):
            # Process chart result
            plot_data = result.get("plotly_graph")
            if plot_data:
                # Convert dictionary to JSON string if needed
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                response_text = "Returning the generated chart."
                # Store the chart
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
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = "Returning the data table."
                # Ensure data_wrangled is a DataFrame
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
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = (
                    "I apologize. There was an issue with generating the chart. "
                    "Returning the data table instead."
                )
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                response_text = (
                    "An error occurred while processing your query. Please try again."
                )
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)