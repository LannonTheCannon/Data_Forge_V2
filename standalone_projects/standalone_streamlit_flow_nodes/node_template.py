import random
from streamlit_flow.elements import StreamlitFlowNode

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

import random
from streamlit_flow.elements import StreamlitFlowNode

class BaseNode:
    def __init__(self, node_id, label, full_question, category, node_type, parent_id=None, color=None, metadata=None):
        self.node_id = node_id
        self.label = label
        self.full_question = full_question
        self.category = category
        self.node_type = node_type  # Must be: "theme", "question", or "terminal"
        self.parent_id = parent_id
        self.color = color or random.choice(COLOR_PALETTE)
        self.metadata = metadata or {}
        self.expanded = False

    def mark_expanded(self):
        self.expanded = True

    def can_expand(self):
        return not self.expanded and self.node_type != "terminal"

    def to_streamlit_node(self):
        return StreamlitFlowNode(
            self.node_id,
            (random.randint(-100, 100), random.randint(-100, 100)),
            {
                "section_path": self.node_id,
                "short_label": self.label,
                "full_question": self.full_question,
                "category": self.category,
                "node_type": self.node_type,
                "metadata": self.metadata,
                "content": f"**{self.label}**"
            },
            "default",
            "right",
            "left",
            style={"backgroundColor": self.color}
        )

    def get_children(self, *args, **kwargs):
        return []


class ThemeNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_children(self, openai_client, metadata_string):
        from question_generator_module import generate_questions_and_labels
        questions, labels = generate_questions_and_labels(
            self.full_question,
            metadata_string,
            mode="theme",
            client=openai_client
        )

        children = []
        for i, (q, lbl) in enumerate(zip(questions, labels), 1):
            node_id = f"{self.node_id}.{i}"
            children.append(QuestionNode(
                node_id=node_id,
                label=lbl,
                full_question=q,
                category=self.category,
                node_type="question",  # ✅ explicitly set
                parent_id=self.node_id
            ))
        return children


class QuestionNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_children(self, openai_client, metadata_string):
        from question_generator_module import generate_questions_and_labels
        questions, labels = generate_questions_and_labels(
            self.full_question,
            metadata_string,
            mode="question",
            client=openai_client
        )

        children = []
        current_depth = self.node_id.count(".")

        for i, (q, lbl) in enumerate(zip(questions, labels), 1):
            node_id = f"{self.node_id}.{i}"

            # 🌱 Make one more generation of questions before terminals
            if current_depth < 2:
                child = QuestionNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="question",
                    parent_id=self.node_id
                )
            else:
                child = TerminalNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="terminal",
                    parent_id=self.node_id
                )

            children.append(child)

        return children



class TerminalNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def can_expand(self):
        return False

    def get_children(self, *args, **kwargs):
        return []