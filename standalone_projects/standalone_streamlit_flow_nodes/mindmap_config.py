# mindmap_config.py
from itertools import cycle
import math, random

CATEGORY_CFG = {
    "Distribution":    {"color": "#4CAF50", "follow": ["Comparison", "Trend", "Outlier"]},
    "Comparison":      {"color": "#2196F3", "follow": ["Correlation", "Drill‑down", "Insight"]},
    "Correlation":     {"color": "#9C27B0", "follow": ["Trend", "Outlier", "Insight"]},
    "Trend":           {"color": "#FF9800", "follow": ["Outlier", "Insight"]},
    "Outlier":         {"color": "#F44336", "follow": ["Drill‑down", "Insight"]},
    "Drill‑down":      {"color": "#795548", "follow": ["Distribution", "Comparison", "Insight"]},
    "Clustering":      {"color": "#3F51B5", "follow": ["Insight"]},
    "Insight":         {"color": "#607D8B", "follow": ["Meta"]},
    "Filtering":       {"color": "#00BCD4", "follow": ["Distribution", "Comparison", "Trend"]},
    "Derived Feature": {"color": "#8BC34A", "follow": ["Distribution", "Correlation", "Insight"]},
    "Sentiment":       {"color": "#FFC107", "follow": ["Trend", "Insight"]},
    "Meta":            {"color": "#9E9E9E", "follow": ["Distribution", "Comparison"]},
}

# -------------------- Novelty-Score Helpers ---------------------#

def _softmax(xs):
    exps = [math.exp(x) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def sample_categories(parent_cat: str, seen: set[str], k_max: int = 3):
    """
    Pick 1‑3 child categories *only* from the parent's `follow` list,
    scoring by novelty within that list.
    """
    prefs = CATEGORY_CFG[parent_cat]["follow"]
    picks, scores = [], []

    for cat in prefs:
        novelty = 2.0 if cat not in seen else 0.3
        picks.append(cat)
        scores.append(novelty)

    # fallback if too few available
    if len(picks) < k_max:
        extras = [c for c in CATEGORY_CFG if c not in picks and c not in seen]
        for cat in extras:
            picks.append(cat)
            scores.append(0.2)
            if len(picks) == k_max:
                break

    probs = _softmax(scores)
    n_children = random.randint(1, min(k_max, len(picks)))
    return random.sample(picks, k=n_children) if len(picks) <= n_children else random.choices(picks, weights=probs, k=n_children)


