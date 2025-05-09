# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-17 19:57:01

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # Ensure the 'Year-Month' column is datetime type
    data_raw['Year-Month'] = pd.to_datetime(data_raw['Year-Month'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data_raw['Year-Month'],
        y=data_raw['Total Amount'],
        mode='lines+markers',
        line=dict(color='royalblue'),
        marker=dict(size=6),
        name='Total Amount'
    ))

    fig.update_layout(
        title='Trend of Total Amount over Year-Month',
        xaxis=dict(
            title='Year-Month',
            type='date',
            tickformat='%Y-%m',
            tickangle=-45,
            showgrid=True
        ),
        yaxis=dict(
            title='Total Amount',
            showgrid=True
        ),
        margin=dict(l=60, r=40, t=80, b=80),
        template='plotly_white'
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict