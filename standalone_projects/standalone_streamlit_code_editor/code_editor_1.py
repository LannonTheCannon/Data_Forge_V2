import streamlit as st
from code_editor import code_editor

# Define the initial code to display in the editor
initial_code = '''def greet(name):
    return f"Hello, {name}!"
'''

# Display the code editor
response = code_editor(initial_code, lang='python', theme='dark', height=300)

# Access the edited code
edited_code = response.get('text', '')

# Optionally, execute the edited code
if st.button("Run Code"):
    exec_locals = {}
    exec(edited_code, {}, exec_locals)
    if 'greet' in exec_locals:
        st.write(exec_locals['greet']("Streamlit User"))