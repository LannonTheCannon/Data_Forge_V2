# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-16 22:04:15

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # Sort the data by movie_count in descending order
    df_sorted = data_raw.sort_values(by='movie_count', ascending=False)

    fig = go.Figure(
        data=go.Bar(
            x=df_sorted['country'],
            y=df_sorted['movie_count'],
            marker_color='indianred'
        )
    )

    fig.update_layout(
        title="Count of Movies Released in Each Country",
        xaxis_title="Country",
        yaxis_title="Movie Count",
        xaxis_tickangle=-45,
        template="plotly_white",
        margin=dict(t=80, b=160)
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict