from langchain_openai import ChatOpenAI
# import pandas as pd
import os
import yaml
import streamlit as st

from IPython.display import Markdown

from ai_data_science_team.agents import DataLoaderToolsAgent

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']

llm = ChatOpenAI(model="gpt-4o-mini")
# llm

# 1.0 Create the data loader agent

# Make a data loader agent
data_loader_agent = DataLoaderToolsAgent(
    llm,
    invoke_react_agent_kwargs={"recursion_limit": 10},
)

# data_loader_agent

# 2.0 Run the Agent!

# data_loader_agent.invoke_agent("What tools do you have access to? Return a table.")
# data_loader_agent.get_ai_message(markdown=True)
#
#
# # Example 2: What folders and files are available?
# data_loader_agent.invoke_agent("What folders and files are available at the root of my directory? Return the file folder structure as code formatted block with the root path at the top and just the top-level folders and files.")
# data_loader_agent.get_ai_message(markdown=True)
# data_loader_agent.get_artifacts(as_dataframe=True)

# Example 3: What is in the data folder?
data_loader_agent.invoke_agent("What is in the data folder?")
data_loader_agent.get_ai_message(markdown=True)
