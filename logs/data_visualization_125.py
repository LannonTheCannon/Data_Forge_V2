# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-21 16:04:22

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import qualitative






    # Define the discovery methods columns and display names
    count_cols = [
        "Count_Astrometry",
        "Count_Disk Kinematics",
        "Count_Eclipse Timing Variations",
        "Count_Imaging",
        "Count_Microlensing",
        "Count_Orbital Brightness Modulation",
        "Count_Pulsar Timing",
        "Count_Pulsation Timing Variations",
        "Count_Radial Velocity",
        "Count_Transit",
        "Count_Transit Timing Variations"
    ]

    display_names = [
        "Astrometry",
        "Disk Kinematics",
        "Eclipse Timing Variations",
        "Imaging",
        "Microlensing",
        "Orbital Brightness Modulation",
        "Pulsar Timing",
        "Pulsation Timing Variations",
        "Radial Velocity",
        "Transit",
        "Transit Timing Variations"
    ]

    # Convert 'Discovery Year' to string for categorical treatment on x-axis
    data = data_raw.copy()
    data['Discovery Year'] = data['Discovery Year'].astype(str)

    # Prepare colors from Plotly qualitative palette (enough distinct colors for 11 categories)
    base_colors = qualitative.Plotly
    if len(base_colors) < len(count_cols):
        base_colors = base_colors + qualitative.Pastel1
    colors = base_colors[:len(count_cols)]

    # Create a stacked bar chart
    fig = go.Figure()

    for col, name, color in zip(count_cols, display_names, colors):
        fig.add_trace(
            go.Bar(
                x=data['Discovery Year'],
                y=data[col],
                name=name,
                marker_color=color,
                hovertemplate='%{x}<br>%{y} discoveries<br>' + name
            )
        )

    fig.update_layout(
        barmode='stack',
        title="Discovery Methods Contribution by Year to Increased Discovery Activity",
        xaxis=dict(
            title="Discovery Year",
            type='category',
            categoryorder='category ascending'
        ),
        yaxis=dict(
            title="Number of Discoveries"
        ),
        legend=dict(
            title="Discovery Method",
            traceorder="normal"
        ),
        margin=dict(t=80, b=50, l=70, r=40)
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict