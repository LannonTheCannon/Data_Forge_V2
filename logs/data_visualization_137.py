# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-26 18:22:08

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.express as px






    # Sort data by Sales Count descending
    df_sorted = data_raw.sort_values(by='Sales Count', ascending=False)

    # Create a bar chart with distinct colors for each product
    fig = go.Figure(
        data=go.Bar(
            x=df_sorted['Product'],
            y=df_sorted['Sales Count'],
            marker=dict(color=px.colors.qualitative.Plotly[:len(df_sorted)]),
        )
    )

    fig.update_layout(
        title="Frequency of Each Product Category Sold",
        xaxis_title="Product",
        yaxis_title="Sales Count",
        xaxis=dict(categoryorder='array', categoryarray=df_sorted['Product']),
        template="plotly_white"
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict