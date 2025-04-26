import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Import those two agents
from ai_data_science_team.agents import DataCleaningAgent
from ai_data_science_team.agents import FeatureEngineeringAgent

# Load Environmental variables
load_dotenv()

# Initailize LLM (Example with OPENAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini')

# Initialize Agents
data_cleaning_agent = DataCleaningAgent(
    model=llm,
    n_samples=50,
    log=False
)

feature_engineering_agent = FeatureEngineeringAgent(
    model=llm,
    n_samples=50,
    log=False
)

# Streamlit app
st.title("🧹🛠️ DataForge - Clean and Engineer Your Dataset!")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read the file into a dataframe
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Tabs Layout
        tabs = st.tabs(["Raw Data", "Cleaned Data", "Feature Engineered Data", "Cleaning Agent Code", "Feature Engineering Code"])

        with tabs[0]:
            st.subheader("STEP 1) Raw Uploaded Data Preview ")
            st.dataframe(df.head())

        # Run Data Cleaning Agent
        with st.spinner('🧹 Cleaning Data...'):
            data_cleaning_agent.invoke_agent(
                data_raw=df,
                user_instructions="Use default cleaning steps."
            )
            df_cleaned = data_cleaning_agent.get_data_cleaned()

        with tabs[1]:
            st.subheader("STEP 2) Cleaned Data Preview")
            st.dataframe(df_cleaned.head())

                # Now run the Feature Engineering Agent
        with st.spinner("Engineering features..."):
            feature_engineering_agent.invoke_agent(
                data_raw=df_cleaned,
                user_instructions="Use default feature engineering steps"
            )
            df_final = feature_engineering_agent.get_data_engineered()

        with tabs[2]:
            st.subheader("STEP 3) Final Feature Engineered Data")
            st.dataframe(df_final.head())

            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download Cleaned Dataset as CSV',
                data=csv,
                file_name='dataforge_cleaned_dataset.csv',
                mime='text/csv'
            )

        with tabs[3]:
            st.subheader(f'STEP 4) Data Cleaning Agent - Generated Code')
            cleaning_code = data_cleaning_agent.get_data_cleaner_function()
            st.code(cleaning_code, language='python')

        with tabs[4]:
            st.subheader('STEP 5) Feature Engineering Agent - Generated Code')
            feature_code = feature_engineering_agent.get_feature_engineer_function()
            st.code(feature_code, language='python')

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")


