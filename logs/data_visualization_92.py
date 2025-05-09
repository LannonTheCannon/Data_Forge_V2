# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-17 02:50:15

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # Ensure year_month is sorted correctly for line plots
    # If year_month is not in datetime format, sort as string (format: YYYY-MM)
    data = data_raw.copy()
    data = data.sort_values('year_month')

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data['year_month'],
            y=data['rating'],
            mode='lines+markers',
            name='Rating'
        )
    )

    fig.update_layout(
        title='Trend of Rating Over Time',
        xaxis_title='Year-Month',
        yaxis_title='Rating',
        xaxis=dict(
            tickmode='array',
            tickvals=data['year_month'][::6],  # Show every 6th tick to avoid clutter
            tickangle=45
        ),
        template='plotly_white'
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict