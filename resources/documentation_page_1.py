import streamlit as st

def documentation_page():
    st.title("📚 Application Documentation")

    with st.expander("🖥️ Application Overview", expanded=True):
        st.markdown("""
        This Streamlit app integrates **PandasAI** and **OpenAI GPT-4** (with Vision) to create an advanced data analysis workflow. It allows users to:

        - **Upload datasets** (CSV/Excel).
        - Generate visualizations using natural-language queries via **PandasAI**.
        - Interpret and analyze these visualizations using GPT-4 Vision.

        The app demonstrates a complete AI-powered data analysis pipeline, focusing on interactive, clear visual insights.
        """)

    with st.expander("📁 Data Upload"):
        st.markdown("""
        Users upload datasets in CSV or Excel formats. Uploaded data is stored in the application's session state.

        **Workflow:**
        ```python
        uploaded_file = st.file_uploader('Upload CSV and Excel Here', type=['csv', 'excel'])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.df_preview = df.head()
            st.session_state.df_summary = df.describe()
        ```

        This step is essential to provide the context for all subsequent analyses.
        """)

    with st.expander("🤖 PandasAI Analysis"):
        st.markdown("""
        Users can ask questions in plain language, and the app translates these into actionable visualizations.

        **Core Components:**
        - **Suggested Questions**: Generated automatically based on dataset metadata.
        - **Natural Language Queries**: Users input questions that are refined by GPT-4 to be visualization-ready.
        ```python
        interpretation = get_assistant_interpretation(user_query, st.session_state['metadata_string'])
        ```
        - **PandasAI**: Converts refined queries into Python code, generates visualizations, and displays the results.
        ```python
        llm = PandasOpenAI(api_token=st.secrets["OPENAI_API_KEY"])
        sdf = SmartDataframe(st.session_state.df, config={"llm": llm})
        answer = sdf.chat(combined_prompt)
        ```

        This interactive component demonstrates a powerful workflow for intuitive data exploration.
        """)

    with st.expander("📊 GPT-4 Vision Interpretation"):
        st.markdown("""
        After generating visualizations, GPT-4 Vision provides detailed interpretations, helping users deeply understand their data.

        **Process:**
        ```python
        result = analyze_chart_with_openai(
            image_path=st.session_state.chart_path,
            user_request=st.session_state.user_query,
            assistant_summary=st.session_state.assistant_interpretation,
            code=st.session_state.pandas_code,
            meta=st.session_state.metadata_string,
        )
        ```

        This component leverages cutting-edge AI capabilities for robust data interpretation.
        """)

    with st.expander("🔧 Custom Callbacks and Utilities"):
        st.markdown("""
        The app employs custom callbacks and utility functions to enhance integration:

        - **StreamlitCallback**: Captures and displays generated Python code from PandasAI.
        ```python
        class StreamlitCallback(BaseCallback):
            def on_code(self, response: str):
                self.generated_code = response
        ```
        - **StreamlitResponse**: Formats dataframes and plots directly into the Streamlit UI.
        ```python
        class StreamlitResponse(ResponseParser):
            def format_dataframe(self, result):
                st.dataframe(result["value"])
        ```
        - **Utility Functions**:
          - `encode_image`: Prepares images for GPT-4 Vision.
          - `get_list_questions`: Auto-generates insightful questions.
          - `get_assistant_interpretation`: Refines user queries.

        These tools create a seamless, interactive experience for users.
        """)

    with st.expander("🎨 UI Customization"):
        st.markdown("""
        Custom CSS and styling provide a visually appealing interface, improving user engagement and readability.

        **Styled Components:**
        ```css
        .big-title {
            font-size:2.0rem;
            font-weight:900;
            color: #2B547E;
        }
        ```

        Streamlit's markdown and CSS capabilities significantly enhance user experience.
        """)

    with st.expander("🚀 Next Steps and Recommendations"):
        st.markdown("""
        Potential future improvements:

        - **Support for larger datasets**: Implement efficient data processing techniques (chunking, lazy loading).
        - **Advanced visualizations**: Integrate interactive libraries like Plotly or Altair.
        - **Enhanced security**: Implement user authentication and secure key storage.

        These enhancements will improve scalability, interactivity, and security for broader adoption.
        """)

    st.markdown("---")
    st.markdown("*Documentation created for ease of understanding the AI-driven data analysis workflow.*")
