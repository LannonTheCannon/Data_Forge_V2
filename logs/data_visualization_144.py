# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-26 22:47:25

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.express as px
    import plotly.io as pio





    # Filter data for the year 2025 (although data_raw only contains 2025)
    df = data_raw.copy()
    df = df[df['Year'] == 2025]

    # Create scatter plot with regression trendline
    fig = px.scatter(
        df,
        x='Total Sales',
        y='Quantity',
        title="Correlation Between Monthly Total Sales and Quantity Sold in 2025",
        labels={"Total Sales": "Total Sales", "Quantity": "Quantity Sold"},
        trendline='ols',
        trendline_color_override='red'
    )

    # Update marker style for better visibility
    fig.update_traces(marker=dict(size=10, symbol='circle', color='blue'), selector=dict(mode='markers'))

    # Ensure regression line is clearly visible and contrasting
    fig.update_traces(selector=dict(mode='lines'), line=dict(width=3, color='red'))

    fig.update_layout(template='plotly_white')

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict