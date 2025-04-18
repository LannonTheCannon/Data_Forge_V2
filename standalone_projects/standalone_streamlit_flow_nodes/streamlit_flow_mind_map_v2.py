import pandas as pd
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import random
import numpy as np
# from standalone_projects.standalone_streamlit_flow_nodes.mindmap_config import sample_categories, CATEGORY_CFG
import openai
from mindmap_config import CATEGORY_CFG, sample_categories
import re

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]


class MMNode(StreamlitFlowNode):
    def __init__(self, node_id, pos, label, full_q, category,
                 payload=None, parent_id=None, status="collapsed"):
        data = {
            "section_path": node_id,
            "short_label": label,
            "full_question": full_q,
            "category": category,
            "payload": payload or {},
            "status": status,
            "content": f"**{label}**"          # ⭐  ADD THIS LINE
        }
        colour = CATEGORY_CFG[category]["color"]
        super().__init__(
            node_id, pos, data,
            "default", "right", "left",
            style={"backgroundColor": colour, "color": "#fff"}
        )


if "curr_state" not in st.session_state:
    root_node = MMNode(
        "S0", (0, 0),
        label="ROOT",
        full_q="",
        category="Meta"
    )
    root_node.data["node_type"] = "root"      # ⭐ so clicks expand later
    root_node.style["backgroundColor"] = COLOR_PALETTE[0]
    st.session_state.curr_state = StreamlitFlowState(nodes=[root_node], edges=[])
    st.session_state.expanded_nodes = set()

for key in ["df", "df_preview", "df_summary", "metadata_string",
            "dataframes", "msg_index", "clicked_questions", "dataset_name", "expanded_nodes"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string",] else []

# --- LLM templates per category --------------------------------------------
PROMPT_BY_CATEGORY: dict[str, str] = {

    # ── 1. DISTRIBUTION ────────────────────────────────────────────────
    "Distribution": """
You are Viz‑GPT.
Return ONE sentence describing the distribution of a single numeric or categorical
column in the dataset, using this template:

    question_text || histogram|bar || x_col || count|none

• Use *histogram* for numeric columns, *bar* for categorical.
• Do **not** add bullet points or extra lines.
• Use only columns present in the data profile.
""",

    # ── 2. COMPARISON ─────────────────────────────────────────────────
    "Comparison": """
You are Viz‑GPT.
Return ONE sentence comparing groups, using this template:

    question_text || grouped_bar|stacked_bar || x_cat_col || y_num_col|agg

Examples of agg: avg, median, sum, count.
One line only, no punctuation before or after the “||” separators.
""",

    # ── 3. CORRELATION & RELATIONSHIP ────────────────────────────────
    "Correlation": """
Return ONE sentence that asks about the relationship between two numeric columns:

    question_text || scatter || x_num_col || y_num_col

If appropriate, mention “with trendline” in the question_text itself.
""",

    # ── 4. TREND ──────────────────────────────────────────────────────
    "Trend": """
Produce ONE sentence about how a metric changes over time:

    question_text || line || date_col || y_num_col|agg
""",

    # ── 5. OUTLIER & ANOMALY ──────────────────────────────────────────
    "Outlier": """
Produce ONE sentence that looks for outliers in a numeric field,
using:

    question_text || box || cat_or_none || num_col
""",

    # ── 6. DRILL‑DOWN ────────────────────────────────────────────────
    "Drill‑down": """
Return ONE sentence that zooms into a subgroup and breaks it down further:

    question_text || bar || sub_cat_col || y_num_col|agg
""",

    # ── 7. CLUSTERING / GROUPING ─────────────────────────────────────
    "Clustering": """
Return ONE sentence that proposes a 2‑D clustering view:

    question_text || scatter || x_num_col || y_num_col
""",

    # ── 8. INSIGHT SUMMARY ───────────────────────────────────────────
    "Insight": """
Return ONE brief narrative bullet:

    question_text || markdown || none || none
Example: "Average rating peaks for light roasts (insight card)"
""",

    # ── 9. FILTERING ─────────────────────────────────────────────────
    "Filtering": """
Return ONE sentence proposing a filter scenario:

    question_text || bar|line|scatter || x_col || y_col|agg
Include the filter in the question_text, e.g. "… for origin = Colombia".
""",

    # ── 10. DERIVED FEATURE ──────────────────────────────────────────
    "Derived Feature": """
Return ONE sentence suggesting a new metric (ratio, bin, etc.):

    question_text || bar|line|scatter || x_newcol || y_col|agg
""",

    # ── 11. SENTIMENT / TEXT ─────────────────────────────────────────
    "Sentiment": """
Return ONE sentence about sentiment trends (if reviews exist):

    question_text || line || date_col || sentiment_score|avg
""",

    # ── 12. META / NEXT‑STEP ─────────────────────────────────────────
    "Meta": """
Suggest ONE next exploration path:

    question_text || markdown || none || none
Example: "Compare rating by roast intensity across countries"
""",
}

THEME_TO_CATEGORY = {
    "Histogram":     "Distribution",
    "Bar Chart":     "Distribution",
    "Scatter Plot":  "Correlation",
    "Box Plot":      "Outlier",
    "Heatmap":       "Correlation",
    "Pivot Table":   "Comparison",
}

THEME_QUESTION = {
    "Histogram":    "What does the distribution of each numeric field look like?",
    "Bar Chart":    "Which categories have the highest counts or averages?",
    "Scatter Plot": "Is there a relationship between two numeric variables?",
    "Box Plot":     "Which groups have unusually high or low values?",
    "Heatmap":      "How are variables correlated with each other?",
    "Pivot Table":  "How do two categorical variables interact?",
}

# ######################### Data Upload function ######################### #

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

def generate_root_summary_question(metadata_string: str) -> str:
    """
    Uses OpenAI to produce a single-sentence question or statement that describes
    or summarizes the dataset metadata.
    :param metadata_str:
    :return: beautified metadata str
    """
    if not metadata_string:
        return "Overview of the dataset"

    messages = [
        {
            "role": "system",
            "content": """You are a data summarizer. 
               Given dataset metadata, produce a single-sentence question 
               or statement that captures the main theme or focus of the dataset. 
               Keep it short (one sentence) and neutral."""
        },
        {
            "role": "user",
            "content": f"Dataset metadata:\n{metadata_string}\n\nPlease provide one short question about the dataset."
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=50  # Enough for a short response
        )
        text = response.choices[0].message.content.strip()
        # Just in case the model returns multiple lines, combine them or take first line
        lines = text.split("\n")
        return lines[0].strip()
    except Exception as e:
        # Fallback if there's an error
        return "What is the primary focus of this dataset?"


# ######################### Mind Mappping function ######################### #

# Not used
def get_list_questions(context: str):
    """
    Generate 4 questions given the dataset metadata plus
    the parent's context (if any).
    """
    if "df_summary" not in st.session_state or st.session_state.df_summary is None:
        return ["No dataset summary found. Please upload a dataset first."]

    combined_context = (
        f"Parent's context/question: {context}\n\n"
        f"Dataset metadata:\n{st.session_state.metadata_string}"
    )

    prompt = """
You are **Viz‑Detective‑GPT**.

Your task is to propose **exactly FOUR** analysis tasks that can each be turned into a
**basic visualisation**.  Follow these rules:

1. **Start with the insight → choose the chart.**  
   Think about the relationship the user might want to see first, then pick
   the simplest chart that reveals it.

2. Stick to these elementary chart families  
   • Histogram (distribution of one numeric column)  
   • Bar chart (count or aggregate of one categorical column)  
   • Grouped / Stacked bar (two categorical columns)  
   • Line chart (trend over an ordered or date column)  
   • Scatter plot (two numeric columns)  
   • Scatter + LOESS / best‑fit line  
   • Box plot (numeric‑by‑categorical)  
   • Heat‑map (correlation or contingency table)  
   • Violin / Strip plot 

   Only use ONE chart if the dataset’s columns make sense for it, create table of 
   INSIGHT if not. Make sure you are explicit in saying "generate a table" if you are look for a key 
   insight for example "What is the highest salary in the dataset" or something like that. 

3. **Column discipline**  
   Use **only** the column names provided in the metadata.  
   Never invent new columns; never rename existing ones.

4. **Output format** – one line per task, no list markers: 

Here's an example 

Create a <chart‑type> to show <insight> using <x_column> on the x‑axis and <y_column> on the y‑axis (<aggregation>).

– Replace `<aggregation>` with avg, sum, count, median, etc., or drop it for raw values.  
– If two columns go on the same axis (e.g. grouped bar), mention both.  
– End the sentence with the proposed chart type in parentheses.

5. **Example** (show the style you must produce):  

Create a grouped bar chart to compare average salary_in_usd for each experience_level across company_size (grouped bar chart).

Return exactly four lines that follow rule 4.           

            """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"Given the context and metadata below, generate 4 short data analysis questions:\n{combined_context}"
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=800
        )
        raw = response.choices[0].message.content.strip()
        questions = raw.split("\n")
        # Filter out empty lines
        cleaned = [q.strip() for q in questions if q.strip()]
        return cleaned

    except Exception as e:
        return [f"Error generating questions: {str(e)}"]

# ++++++++++++++++++++++++ Generate Multiple Questions Sub Function +++++++++++++++++++++++++ #

# Not used
def paraphrase_questions(questions):
    """
    Paraphrase multiple questions into short labels or titles.
    Returns a list of short strings (one for each question).
    """
    joined = "\n".join(questions)
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that transforms full-length data analysis
            questions into short, descriptive labels (5-8 words max). 
            Example: 'How do sales vary by region?' -> 'Sales by Region'."""
        },
        {
            "role": "user",
            "content": f"Please paraphrase each question into a short label:\n{joined}"
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        # Attempt to match them one-to-one
        # Ensure we only return as many paraphrases as we had questions
        paraphrased = lines[:len(questions)]
        return paraphrased

    except Exception as e:
        # On error, just return the original questions
        return questions

# Not Used
def get_section_path_children(parent_path: str, num_children=4):
    """
    Given a parent's section path like 'S0.1', produce a list:
    ['S0.1.1', 'S0.1.2', 'S0.1.3', 'S0.1.4'].
    """
    children = []
    for i in range(1, num_children + 1):
        new_path = f"{parent_path}.{i}"
        children.append(new_path)
    return children

# Not Used
def get_color_for_depth(section_path: str):
    """
    Depth = number of dots in the section path
    For example, S0 -> depth 0, S0.1 -> depth 1, S0.1.2 -> depth 2, etc.
    Then use your existing COLOR_PALETTE.
    """
    depth = section_path.count(".")
    return COLOR_PALETTE[depth % len(COLOR_PALETTE)]


def llm_generate(category: str, df_profile: str) -> tuple[str, str]:
    """
    Returns
    -------
    full_question : str   # copy‑pastable prompt for the viz agent
    payload       : str   # “scatter | x=price | y=rating”
    """
    sys_tmpl = PROMPT_BY_CATEGORY[category]
    user_msg = f"DATA PROFILE:\n{df_profile}\n\nUse only columns that exist."

    def _ask(system_prompt: str) -> str:
        resp = openai.chat.completions.create(
            model="gpt-4.1-nano",
            temperature=0.25,
            max_tokens=120,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_msg}],
        )
        return resp.choices[0].message.content.splitlines()[0].strip()

    # ── 1st try -------------------------------------------------------------
    line = _ask(sys_tmpl)

    # if the “question_text” looks like a lone column name → try again once
    if "||" not in line or line.count("||") < 3 or len(line.split(" ")[0]) < 4:
        repair_prompt = (
            sys_tmpl
            + "\n\nYour previous answer was not a full question. "
            + "Rewrite it so that the part before the first '||' is a clear English question."
        )
        line = _ask(repair_prompt)

    # ── attempt to parse ----------------------------------------------------
    parts = [p.strip() for p in line.split("||")]
    if len(parts) == 4:
        q_txt, chart, x, y = parts
        payload = f"{chart} | x={x} | y={y}"
        return q_txt, payload

    # ── ultimate fall‑back --------------------------------------------------
    # ➊ pick two columns that exist
    num_cols = [c for c in re.findall(r"`([^`]+)`", df_profile)] or ["col1", "col2"]
    x = num_cols[0]
    y = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    q_txt = f"What is the relationship between {x} and {y}?"
    payload = f"scatter | x={x} | y={y}"
    return q_txt, payload

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Expanding Node with Questions Function XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #


def expand_root_node(clicked_node):
        insight_types = [
            ("Explore Distributions", "Distribution"),
            ("Compare Groups", "Comparison"),
            ("Find Relationships", "Correlation"),
            ("Detect Trends", "Trend"),
            ("Spot Outliers", "Outlier"),
            ("Suggest a Starting Point", "Meta"),
        ]

        parent_path = clicked_node.data["section_path"]

        for idx, (label, cat) in enumerate(insight_types, start=1):
            child_path = f"{parent_path}.{idx}"
            color = CATEGORY_CFG[cat]["color"]
            full_q = f"Would you like to {label.lower()} in the data?"

            new_node = MMNode(
                child_path,
                (random.randint(-120, 120), random.randint(-120, 120)),
                label=label,
                full_q=full_q,
                category=cat,
                parent_id=clicked_node.id
            )
            new_node.data["node_type"] = "intent"
            new_node.style["backgroundColor"] = color

            st.session_state.curr_state.nodes.append(new_node)
            st.session_state.curr_state.edges.append(
                StreamlitFlowEdge(f"{clicked_node.id}-{child_path}", clicked_node.id, child_path, animated=True)
            )

        st.session_state.expanded_nodes.add(clicked_node.id)

# def expand_node_with_questions(clicked_node):
#     """
#     Expand any thematic node into 4 EDA‑style questions that
#     consider both numeric and categorical columns.
#     """
#     # 1) Determine theme (e.g. "Histogram", "Bar Chart", etc.)
#     theme = clicked_node.data.get("full_question", "")
#
#     context = f"Theme: {theme}\n\nDataset metadata:\n{st.session_state.metadata_string}"
#     q1 = get_list_questions(context)
#     short_labels = paraphrase_questions(q1)
#
#     # 5) Create child nodes for each question
#     parent_path = clicked_node.data.get("section_path", "S0")
#     child_paths = get_section_path_children(parent_path, num_children=len(q1))
#     for i, child_path in enumerate(child_paths):
#         full_q = q1[i]
#         label = short_labels[i]
#         color = get_color_for_depth(child_path)
#
#         node_data = {
#             "section_path": child_path,
#             "short_label": label,
#             "full_question": full_q,
#             "content": f"**{label}**"
#         }
#
#         new_node = StreamlitFlowNode(
#             child_path,
#             (random.randint(-100, 100), random.randint(-100, 100)),
#             node_data,
#             "default",
#             "right",
#             "left",
#             style={"backgroundColor": color}
#         )
#         st.session_state.curr_state.nodes.append(new_node)
#         st.session_state.curr_state.edges.append(
#             StreamlitFlowEdge(f"{clicked_node.id}-{child_path}", clicked_node.id, child_path, animated=True)
#         )
#
#     # 6) Mark as expanded & log the click
#     st.session_state.expanded_nodes.add(clicked_node.data["section_path"])
#     if parent_path not in {q["section"] for q in st.session_state.clicked_questions}:
#         st.session_state.clicked_questions.append({
#             "section": parent_path,
#             "short_label": clicked_node.data.get("short_label", parent_path),
#             "full_question": theme
#         })

def expand_node(clicked_node):
    """Expand any non‑root node according to smart category logic."""
    parent_cat = clicked_node.data.get("category", "Meta")
    parent_path = clicked_node.data["section_path"]

    MAX_DEPTH = 4
    if parent_path.count(".") >= MAX_DEPTH:
        st.warning("Max depth reached for this branch.")
        return

    # track what cats we've already spawned *under this parent*
    already = st.session_state.get(f"seen_{parent_path}", set())

    # 1) pick 1‑3 child categories with novelty‑weighted sampling
    child_cats = sample_categories(parent_cat, already)

    # 2) LLM → concrete Q for each child category
    children = []
    for cat in child_cats:
        q_txt, payload = llm_generate(cat, st.session_state.metadata_string)
        label = paraphrase_questions([q_txt])[0]
        # label = q_txt.split("?")[0][:30].strip()
        node_id = f"{parent_path}.{len(children)+1}"
        n = MMNode(node_id,
                   (random.randint(-120, 120), random.randint(-120, 120)),
                   label, q_txt, cat, payload, parent_id=clicked_node.id)
        children.append(n)

    # 3) add to flow graph
    for n in children:
        st.session_state.curr_state.nodes.append(n)
        st.session_state.curr_state.edges.append(
            StreamlitFlowEdge(f"{clicked_node.id}-{n.id}", clicked_node.id, n.id, animated=True)
        )

    # 4) mark bookkeeping
    st.session_state.expanded_nodes.add(clicked_node.id)
    already.update(child_cats)
    st.session_state[f"seen_{parent_path}"] = already

def log_click(node: MMNode):
    path = node.data["section_path"]
    if any(r["section_path"] == path for r in st.session_state.get("clicked_questions", [])):
        return

    st.session_state.clicked_questions.append({
        "section_path": path,
        "short_label" : node.data["short_label"],
        "full_question": node.data["full_question"],
        "category"    : node.data["category"],
        "payload"     : node.data["payload"],
    })

PAGE_OPTIONS = [
    'Data Upload',
    'Mind Mapping',
]

page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('Upload your own Dataset!')
        uploaded_file = st.file_uploader('Upload CSV or Excel here', type=['csv', 'excel'])

        if uploaded_file is not None:
            # Load data into session state
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state["DATA_RAW"] = df
                st.session_state.df_preview = df.head()
                # st.session_state.df_summary = df.describe()

                # Save dataset name without extension
                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name
                # st.write(dataset_name)


                # numeric + categorical summary
                numeric_summary = df.describe()
                cat_summary = df.describe(include=['object', 'category', 'bool'])
                # build a richer metadata string
                st.session_state.df_summary = numeric_summary  # keep for display
                cols = df.columns.tolist()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                cat_cardinalities = {c: int(df[c].nunique()) for c in categorical_cols}
                top_cats = {c: df[c].value_counts().head(3).to_dict() for c in categorical_cols}

                # Rebuild a 'metadata_string' for the root node
                if st.session_state.df_summary is not None:
                    # Basic example of turning summary + columns into a string
                    cols = list(st.session_state.df_summary.columns)
                    row_count = st.session_state.df.shape[0]
                    st.session_state.metadata_string = (
                        f"Columns: {cols}\n"
                        f"Numeric columns: {numeric_cols}\n"
                        f"Categorical columns: {categorical_cols} (cardinalities: {cat_cardinalities})\n"
                        f"Top categories: {top_cats}\n"
                        f"Row count: {len(df)}"
                    )
                    # print(st.session_state.metadata_string)
                    # Produce a one-sentence question describing the dataset
                    root_question = generate_root_summary_question(st.session_state.metadata_string)

                    # Update the root node's data if it exists:
                    if st.session_state.curr_state.nodes:
                        root_node = st.session_state.curr_state.nodes[0]
                        root_node.data["full_question"] = root_question
                        # Optionally display it on the node itself:
                        root_node.data["content"] = "ROOT"  # or root_node.data["content"] = root_question
                        root_node.data["short_label"] = "ROOT"

        # Display preview & summary if data exists
        if st.session_state.df is not None:
            st.write("### Data Preview")
            st.write(st.session_state.df_preview)

            st.write("### Data Summary")
            st.write(st.session_state.df_summary)


    elif page == "Mind Mapping":
        st.title("Mind Mapping + Agentic Ensemble")
        # ── keep the root’s label in sync with the uploaded file name
        if st.session_state.get("dataset_name"):
            root_node = st.session_state.curr_state.nodes[0]
            if root_node.data["content"] != st.session_state["dataset_name"]:
                root_node.data["content"] = st.session_state["dataset_name"]
        # ── reset button ---------------------------------------------------------
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Reset Mind Map"):
                dataset_label = st.session_state.get("dataset_name", "Dataset")
                new_root = MMNode(
                    "S0", (0, 0),
                    label="ROOT",
                    full_q=st.session_state.metadata_string or "",
                    category="Meta",
                )
                new_root.data["node_type"] = "root"
                new_root.style["backgroundColor"] = COLOR_PALETTE[0]
                st.session_state.curr_state = StreamlitFlowState(nodes=[new_root], edges=[])
                st.session_state.expanded_nodes = set()
                st.session_state.clicked_questions = []
                st.rerun()
        # ── render the flow ------------------------------------------------------
        st.session_state.curr_state = streamlit_flow(
            "mind_map",
            st.session_state.curr_state,
            layout=TreeLayout(direction="right"),
            fit_view=True,
            height=550,
            get_node_on_click=True,
            enable_node_menu=True,
            enable_edge_menu=True,
            show_minimap=False,
        )

        # ── handle clicks --------------------------------------------------------
        clicked_id = st.session_state.curr_state.selected_id
        if clicked_id and clicked_id not in st.session_state.expanded_nodes:
            node_map = {n.id: n for n in st.session_state.curr_state.nodes}
            clicked_node = node_map.get(clicked_id)
            if clicked_node:  # log once per unique click
                log_click(clicked_node)
                node_type = clicked_node.data.get("node_type", "")
                if node_type == "root":
                    expand_root_node(clicked_node)  # adds 6 theme boxes
                else:
                    expand_node(clicked_node)  # recursive children
                # clear selection & re‑render
                st.session_state.curr_state.selected_id = None
                st.rerun()

        # ── travel log -----------------------------------------------------------
        if st.session_state.clicked_questions:
            st.write("### Exploration Path")
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            st.table(df_log)
