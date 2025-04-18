def generate_questions_and_labels(prompt: str, metadata_string: str, mode="theme", client=None):
    """
    Given a prompt and metadata, return 4 chart-based questions, short labels, and node_types.
    Node types can be 'question' or 'terminal' to control tree depth.
    """
    if not client:
        raise ValueError("You must provide an OpenAI client.")

    # Determine instruction set
    if mode == "theme":
        context = f"Theme: {prompt}\n\nDataset metadata:\n{metadata_string}"
        instruction = "Propose 4 EDA questions based on this theme."
    else:
        context = f"Follow-up for: {prompt}\n\nDataset metadata:\n{metadata_string}"
        instruction = "Propose 4 follow-up analysis questions that may lead to insights."

    # 🔍 Step 1 — Get the questions
    question_prompt = f"""
{instruction}

You are **Viz‑Detective‑GPT**, a data analyst AI.

Your task:
- Propose **exactly 4** visual analysis tasks (chart-based).
- Each task should be 1 sentence long and explain the goal of a chart.

Allowed chart types:
- Histogram, Bar chart, Grouped / Stacked bar, Line chart, Scatter plot, Scatter + best-fit line, Box plot, Heat-map, Table

Use only column names from metadata. Avoid made-up columns.

Each line = a full sentence. No list markers.

Example:
Create a grouped bar chart to compare average salary for each role across industries (grouped bar chart).
"""

    messages = [
        {"role": "system", "content": question_prompt},
        {"role": "user", "content": context}
    ]

    try:
        question_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=600
        )
        raw_questions = question_response.choices[0].message.content.strip().split("\n")
        questions = [q.strip() for q in raw_questions if q.strip()]
    except Exception as e:
        return [f"Error: {e}"], [f"Label {i+1}" for i in range(4)], ["terminal"] * 4

    # ✏️ Step 2 — Paraphrase into labels
    label_messages = [
        {"role": "system", "content": "Rewrite each question as a short chart title (4–8 words max)."},
        {"role": "user", "content": "\n".join(questions)}
    ]

    try:
        label_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=label_messages,
            max_tokens=300
        )
        raw_labels = label_response.choices[0].message.content.strip().split("\n")
        labels = [l.strip() for l in raw_labels if l.strip()]
    except Exception:
        labels = [f"Q{i+1}" for i in range(len(questions))]

    # 🧠 Step 3 — Classify as 'question' or 'terminal'
    type_prompt = """
For each of the following analysis questions, classify whether it's a broad exploratory *question* or a specific *terminal* insight.

Output exactly one word per line: either 'question' or 'terminal'.

Definitions:
- 'question': leads to follow-up exploration (e.g., comparisons, breakdowns)
- 'terminal': answers a specific insight with a clear chart or conclusion
"""
    type_messages = [
        {"role": "system", "content": type_prompt},
        {"role": "user", "content": "\n".join(questions)}
    ]

    try:
        type_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=type_messages,
            max_tokens=200
        )
        raw_types = type_response.choices[0].message.content.strip().split("\n")
        node_types = [t.strip().lower() for t in raw_types if t.strip().lower() in {"question", "terminal"}]
    except Exception:
        node_types = ["terminal"] * len(questions)

    # Fallback if lengths mismatch
    while len(labels) < len(questions):
        labels.append(f"Q{len(labels) + 1}")
    while len(node_types) < len(questions):
        node_types.append("terminal")

    return questions, labels, node_types
