# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-16 22:22:31

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # We expect data_raw as a DataFrame containing columns:
    # genre, q25_revenue, median_revenue, q75_revenue

    fig = go.Figure()

    # For each revenue type, add a box trace grouped by genre
    # But a box plot normally shows distribution for each group.
    # Here, we have only 1 value per genre for each q25, median, q75.
    # So to create meaningful box plots, we can create three box plots per genre
    # but since each is a single value, this is not typical usage.
    # Instead, we can create 3 box plots: one for q25_revenue across genres,
    # one for median_revenue, one for q75_revenue to compare distributions across genres,
    # or create a grouped box plot with genre on x and y values from q25, median, q75 per genre.
    # But the instruction is to compare distribution of revenue across genres with box plots
    # using x=genre and y=[q25_revenue, median_revenue, q75_revenue]
    # This suggests we should show three box plots per genre showing the distribution of these values.
    #
    # Because each genre has only one row, we cannot create meaningful box plots per genre.
    # Instead, create three box plots, one per revenue quantile, grouping by genre on x-axis.
    #
    # But box plots require multiple data points per category.
    # Since data has only one value per genre per revenue quantile,
    # a box plot per genre is not feasible.
    #
    # Alternative is to plot three box plots: one for q25_revenue values across genres,
    # one for median_revenue values across genres, one for q75_revenue values across genres,
    # with x as the revenue type.
    #
    # However, the instruction is explicit: x=genre, y=[q25_revenue, median_revenue, q75_revenue].
    # So, we will create a box plot per genre, where each box plot contains the three values
    # q25_revenue, median_revenue, q75_revenue as separate points to form a box.
    #
    # Since only three points per genre, it forms a minimal box plot.
    # This will visualize revenue distribution per genre as a box defined by these three quantiles.
    #
    # So for each genre, create a box plot with these three points:
    # q25_revenue, median_revenue, q75_revenue

    # Prepare data for box plots:
    # We will create one box trace per genre with the three y values as the data points.

    for i, row in data_raw.iterrows():
        y_vals = [row['q25_revenue'], row['median_revenue'], row['q75_revenue']]
        fig.add_trace(go.Box(
            y=y_vals,
            name=row['genre'],
            boxpoints='all',
            jitter=0.3,
            whiskerwidth=0,
            marker_size=4,
            line_width=1,
            showlegend=False
        ))

    fig.update_layout(
        title="Box Plot Comparing Distribution of Revenue Across Different Genres",
        yaxis_title="Revenue Distribution",
        xaxis_title="Genre",
        xaxis=dict(tickangle=45),
        yaxis_type="linear",
        boxmode='group',
        margin=dict(b=150),
        height=600,
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict