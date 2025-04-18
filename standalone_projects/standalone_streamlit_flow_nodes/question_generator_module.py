def generate_questions_and_labels(prompt: str, metadata_string: str, mode="theme", client=None):
    """
    Given a prompt and metadata, return 4 chart-based questions and paraphrased labels.
    """
    if not client:
        raise ValueError("You must provide an OpenAI client.")

    # Determine instruction set
    if mode == "theme":
        context = f"Theme: {prompt}\n\nDataset metadata:\n{metadata_string}"
        instruction = "Propose 4 EDA questions based on this theme."
    else:
        context = f"Follow-up for: {prompt}\n\nDataset metadata:\n{metadata_string}"
        instruction = "Propose 4 follow-up analysis questions."

    question_prompt = f"""
{instruction}

You are **Viz‑Detective‑GPT**. Your task is to write **exactly four** analysis tasks that can each be turned into a **basic visualisation**.

Use ONLY these chart types:
- Histogram
- Bar chart
- Grouped / Stacked bar
- Line chart
- Scatter plot
- Scatter + best‑fit line
- Box plot
- Heat‑map
- Table (for non-visual answers)

Use ONLY the column names provided in the metadata.

Output **one complete sentence per line** with no list markers.

Example:  
Create a grouped bar chart to compare average salary for each role across industries (grouped bar chart).
"""

    messages = [
        {"role": "system", "content": question_prompt},
        {"role": "user", "content": context}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=600
        )
        raw_questions = response.choices[0].message.content.strip().split("\n")
        questions = [q.strip() for q in raw_questions if q.strip()]
    except Exception as e:
        return [f"Error: {e}"], [f"Label {i+1}" for i in range(4)]

    if len(questions) < 4:
        return questions, [f"Q{i+1}" for i in range(len(questions))]

    # Paraphrase into short labels
    label_messages = [
        {"role": "system", "content": "Rewrite each question as a short chart title (4-8 words max)."},
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

    # Return only as many labels as questions
    labels = labels[:len(questions)]
    return questions, labels
