import os
import time
import base64
import numpy as np
import shutil
import io
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_ace import st_ace
import inspect
from streamlit_elements import elements, dashboard, mui, html
import json
from code_editor import code_editor
import traceback
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.callbacks import BaseCallback
from pandasai.responses.response_parser import ResponseParser
import plotly.graph_objects as go
import plotly.io as pio
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout
import random
from uuid import uuid4
import streamlit.components.v1 as components
from pathlib import Path
import html
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent

# Fallback for missing functions in ai_data_science_team.templates
try:
    from ai_data_science_team.templates import format_agent_name, format_recommended_steps
except ImportError:
    def format_agent_name(agent_name):
        return f"Agent: {agent_name}"


    def format_recommended_steps(steps, heading=""):
        return heading + "\n" + steps

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =============================================================================
# CUSTOM AGENT EXTENSION
# =============================================================================
# We subclass DataVisualizationAgent to override its _make_compiled_graph() method.
# Our custom function (make_custom_data_visualization_agent) is nearly identical to the
# original, except in the chart_generator node we add explicit instructions that force
# the LLM to use the exact column names from the provided data summary.

class CustomDataVisualizationAgent(DataVisualizationAgent):
    def _make_compiled_graph(self):
        self.response = None
        return make_custom_data_visualization_agent(**self._params)


def make_custom_data_visualization_agent(
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_visualization.py",
        function_name="data_visualization",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer=None,
):
    llm = model

    if human_in_the_loop:
        if checkpointer is None:
            print("Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver().")
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()

    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")

    if log:
        if log_path is None:
            log_path = os.path.join(os.getcwd(), "logs/")
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # -------------------------------------------------------------------------
    # Define GraphState for the routing process
    # -------------------------------------------------------------------------
    from typing import TypedDict, Annotated, Sequence, Literal
    import operator
    from langchain_core.messages import BaseMessage
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        user_instructions_processed: str
        recommended_steps: str
        data_raw: dict
        plotly_graph: dict
        all_datasets_summary: str
        data_visualization_function: str
        data_visualization_function_path: str
        data_visualization_function_file_name: str
        data_visualization_function_name: str
        data_visualization_error: str
        max_retries: int
        retry_count: int

    # -------------------------------------------------------------------------
    # chart_instructor Node (unchanged)
    # -------------------------------------------------------------------------
    from ai_data_science_team.tools.dataframe import get_dataframe_summary

    def chart_instructor(state: GraphState):
        print(format_agent_name("data_visualization_agent"))
        print("    * CREATE CHART GENERATOR INSTRUCTIONS")
        from langchain.prompts import PromptTemplate
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 

            You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart.

            USER QUESTION / INSTRUCTIONS: 
            {user_instructions}

            Previously Recommended Instructions (if any):
            {recommended_steps}

            DATA SUMMARY: 
            {all_datasets_summary}

            IMPORTANT:

            - Formulate chart generator instructions by informing the chart generator of what type of plotly plot to use.
            - Select the appropriate chart type based on the data summary and user’s question.
            - Provide an informative title, X and Y axis titles.

            RETURN FORMAT:

            Return your instructions in the following format:
            CHART GENERATOR INSTRUCTIONS: 
            FILL IN THE INSTRUCTIONS HERE

            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated instructions.
            """,
            input_variables=[
                "user_instructions",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples, skip_stats=False)
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        chart_instructor_prompt = recommend_steps_prompt | llm
        recommended_steps = chart_instructor_prompt.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
            }
        )
        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Data Cleaning Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    # -------------------------------------------------------------------------
    # Custom chart_generator Node with explicit instructions on column names
    # -------------------------------------------------------------------------
    def chart_generator(state: GraphState):
        print("    * CUSTOM CREATE DATA VISUALIZATION CODE")
        if bypass_recommended_steps:
            print(format_agent_name("data_visualization_agent"))
            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)
            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples, skip_stats=False)
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
            chart_generator_instructions = state.get("user_instructions")
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            chart_generator_instructions = state.get("recommended_steps")

        from langchain.prompts import PromptTemplate
        prompt_template = PromptTemplate(
            template="""
            You are a chart generator agent that is an expert in generating plotly charts. You must use plotly or plotly.express to produce plots.

            Your job is to produce python code to generate visualizations with a function named {function_name}.

            You will take instructions from a Chart Instructor and generate a plotly chart from the data provided.

            CHART INSTRUCTIONS: 
            {chart_generator_instructions}

            DATA: 
            {all_datasets_summary}

            IMPORTANT:
            The DATA section above lists the exact column names present in the dataset.
            You MUST use these names exactly as they appear.
            Do NOT substitute or alias them (for example, do not change 'salary_in_usd' to 'average_salary_proxy').

            RETURN:

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports.

            Return the plotly chart as a dictionary.

            Return code to provide the data visualization function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                import json
                import plotly.graph_objects as go
                import plotly.io as pio

                ...

                fig_json = pio.to_json(fig)
                fig_dict = json.loads(fig_json)

                return fig_dict

            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated instructions.

            """,
            input_variables=[
                "chart_generator_instructions",
                "all_datasets_summary",
                "function_name",
            ],
        )
        from ai_data_science_team.parsers.parsers import PythonOutputParser
        data_visualization_agent = prompt_template | llm | PythonOutputParser()
        response = data_visualization_agent.invoke(
            {
                "chart_generator_instructions": chart_generator_instructions,
                "all_datasets_summary": all_datasets_summary_str,
                "function_name": function_name,
            }
        )
        from ai_data_science_team.utils.regex import relocate_imports_inside_function, add_comments_to_top
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name="data_visualization_agent")
        from ai_data_science_team.utils.logging import log_ai_function
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )
        return {
            "data_visualization_function": response,
            "data_visualization_function_path": file_path,
            "data_visualization_function_file_name": file_name_2,
            "data_visualization_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
        }

    # -------------------------------------------------------------------------
    # Human Review, Execute, Fix and Report Nodes (same as original)
    # -------------------------------------------------------------------------
    prompt_text_human_review = "Are the following data visualization instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    from ai_data_science_team.templates import node_func_human_review, node_func_execute_agent_code_on_data, \
        node_func_fix_agent_code, node_func_report_agent_outputs

    if not bypass_explain_code:
        def human_review(state: GraphState):
            from langgraph.types import Command
            from typing import Literal
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="explain_data_visualization_code",
                no_goto="chart_instructor",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_visualization_function",
            )
    else:
        def human_review(state: GraphState):
            from langgraph.types import Command
            from typing import Literal
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="chart_instructor",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_visualization_function",
            )

    def execute_data_visualization_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="plotly_graph",
            error_key="data_visualization_error",
            code_snippet_key="data_visualization_function",
            agent_function_name=state.get("data_visualization_function_name"),
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            error_message_prefix="An error occurred during data visualization: ",
        )

    def fix_data_visualization_code(state: GraphState):
        prompt = """
        You are a Data Visualization Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is broken and must be fixed.

        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), including all imports.

        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """
        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_visualization_function",
            error_key="data_visualization_error",
            llm=llm,
            prompt_template=prompt,
            agent_name="data_visualization_agent",
            log=log,
            file_path=state.get("data_visualization_function_path"),
            function_name=state.get("data_visualization_function_name"),
        )

    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_visualization_function",
                "data_visualization_function_path",
                "data_visualization_function_name",
                "data_visualization_error",
            ],
            result_key="messages",
            role="data_visualization_agent",
            custom_title="Data Visualization Agent Outputs",
        )

    node_functions = {
        "chart_instructor": chart_instructor,
        "human_review": human_review,
        "chart_generator": chart_generator,
        "execute_data_visualization_code": execute_data_visualization_code,
        "fix_data_visualization_code": fix_data_visualization_code,
        "report_agent_outputs": report_agent_outputs,
    }

    from ai_data_science_team.templates import create_coding_agent_graph
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="chart_instructor",
        create_code_node_name="chart_generator",
        execute_code_node_name="execute_data_visualization_code",
        fix_code_node_name="fix_data_visualization_code",
        explain_code_node_name="report_agent_outputs",
        error_key="data_visualization_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name="data_visualization_agent",
    )

    return app


# =============================================================================
# END OF CUSTOM AGENT EXTENSION
# =============================================================================

# =============================================================================
# STREAMLIT APPLICATION CODE
# =============================================================================
for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string",
                                                "DATA_RAW"] else []


def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    return None


def display_chat_history():
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}
    for i, msg in enumerate(msgs.messages):
        role_label = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role_label}:** {msg.content}")
            if i in st.session_state["chat_artifacts"]:
                for j, artifact in enumerate(st.session_state["chat_artifacts"][i]):
                    with st.expander(f"\U0001F4CE {artifact['title']}", expanded=True):
                        tabs = st.tabs(["\U0001F4CA Output", "💻 Code", "User Output"])
                        unique_key = f"msg_{i}_artifact_{j}"
                        with tabs[0]:
                            render_type = artifact.get("render_type")
                            data = artifact.get("data")
                            if isinstance(data, dict) and "data" in data and "layout" in data:
                                data = pio.from_json(json.dumps(data))
                            if render_type == "plotly":
                                st.plotly_chart(
                                    data,
                                    use_container_width=True,
                                    config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
                                    key=f"plotly_chart_{unique_key}_original"
                                )
                            elif render_type == "dataframe":
                                st.dataframe(data, key=f"dataframe_{unique_key}_original")
                            elif render_type == "string":
                                st.markdown(data, key=f"string_{unique_key}_original")
                            elif render_type == "number":
                                st.write(data, key=f"number_{unique_key}_original")
                            else:
                                st.warning("Unknown artifact type.", key=f"warning_{unique_key}_original")
                        with tabs[1]:
                            editor_response = code_editor(
                                code=artifact.get("code", "# No code available"),
                                lang="python",
                                theme="dracula",
                                height=300,
                                buttons=[{
                                    "name": "Run",
                                    "feather": "Play",
                                    "primary": True,
                                    "hasText": True,
                                    "showWithIcon": True,
                                    "commands": ["submit"],
                                    "style": {"bottom": "0.44rem", "right": "0.4rem"}
                                }],
                                key=f"code_editor_{unique_key}"
                            )
                            code_to_run = editor_response.get("text", "").strip()
                        with tabs[2]:
                            if st.button("Run Code", key=f"run_code_{unique_key}"):
                                if not code_to_run:
                                    st.error("No code to run. Please check your code.")
                                else:
                                    try:
                                        exec_globals = {
                                            "df": st.session_state.df,
                                            "pd": pd,
                                            "np": np,
                                            "sns": sns,
                                            "go": go,
                                            "plt": plt,
                                            "pio": pio,
                                            "st": st
                                        }
                                        exec_locals = {}
                                        exec(code_to_run, exec_globals, exec_locals)
                                        output_obj = None
                                        if "fig" in exec_locals:
                                            output_obj = exec_locals["fig"]
                                        elif "output" in exec_locals:
                                            output_obj = exec_locals["output"]
                                        elif "fig_dict" in exec_locals:
                                            output_obj = exec_locals["fig_dict"]
                                        if output_obj is None and "data_visualization" in exec_locals:
                                            output_obj = exec_locals["data_visualization"](st.session_state.df)
                                        if output_obj is None:
                                            st.error(
                                                "No output detected. Ensure your code assigns the output or defines a function 'data_visualization(data_raw)'.")
                                        else:
                                            if isinstance(output_obj,
                                                          dict) and "data" in output_obj and "layout" in output_obj:
                                                output_obj = pio.from_json(json.dumps(output_obj))
                                            if isinstance(output_obj, go.Figure):
                                                st.plotly_chart(
                                                    output_obj,
                                                    use_container_width=True,
                                                    config={"displayModeBar": True, "scrollZoom": True,
                                                            "displaylogo": False},
                                                    key=f"plotly_chart_{unique_key}_useroutput"
                                                )
                                            elif isinstance(output_obj, pd.DataFrame):
                                                st.dataframe(output_obj, key=f"dataframe_{unique_key}_useroutput")
                                            else:
                                                st.write(output_obj, key=f"output_{unique_key}_useroutput")
                                    except Exception as e:
                                        st.error(f"Error executing code: {e}")


PAGE_OPTIONS = ['Data Upload', 'Data Analyst', 'Data Storytelling']
page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

if __name__ == "__main__":
    if page == 'Data Upload':
        st.title('Upload your own Dataset!')
        uploaded_file = st.file_uploader('Upload CSV or Excel here', type=['csv', 'excel'])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state["DATA_RAW"] = df
                st.session_state.df_preview = df.head()
                st.session_state.df_summary = df.describe()
                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name
                st.write(dataset_name)
                if st.session_state.df_summary is not None:
                    cols = list(st.session_state.df_summary.columns)
                    row_count = st.session_state.df.shape[0]
                    st.session_state.metadata_string = (
                        f"Columns: {cols}\nTotal Rows: {row_count}\nSummary Stats:\n{st.session_state.df_summary}"
                    )
            if st.session_state.df is not None:
                st.write("### Data Preview")
                st.write(st.session_state.df_preview)
                st.write("### Data Summary")
                st.write(st.session_state.df_summary)
    elif page == 'Data Analyst':
        st.subheader('Pandas Data Analyst Mode')
        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
        if len(msgs.messages) == 0:
            msgs.add_ai_message(
                "IMPORTANT: The DATA section below lists the exact column names. Use these names exactly without substituting defaults.")
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4.1-nano', api_key=st.secrets['OPENAI_API_KEY'])
            st.session_state.pandas_data_analyst = PandasDataAnalyst(
                model=model,
                data_wrangling_agent=DataWranglingAgent(model=model, log=False, n_samples=100),
                data_visualization_agent=CustomDataVisualizationAgent(model=model, log=False, n_samples=100,
                                                                      bypass_recommended_steps=True)
            )
        question = st.chat_input('Ask a question about your dataset!')
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
                        st.session_state["chat_artifacts"][msg_index].append({
                            "title": "Chart",
                            "render_type": "plotly",
                            "data": plot_obj,
                            "code": result.get("data_visualization_function")
                        })
                    elif route == "table":
                        df = result.get("data_wrangled")
                        if df is not None:
                            st.session_state.dataframes.append(df)
                            st.session_state["chat_artifacts"][msg_index].append({
                                "title": "Table",
                                "render_type": "dataframe",
                                "data": df,
                                "code": result.get("data_wrangler_function")
                            })
                except Exception as e:
                    msgs.add_ai_message(f"Error: {e}")
        display_chat_history()
    else:
        st.info("Data Storytelling page coming soon!")
