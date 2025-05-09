# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-28 17:07:53

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    df = data_raw.copy()

    # To simulate box plots from summary statistics, we need:
    # min, Q1, median, Q3, max
    # Given min, median, max, std, we approximate Q1 and Q3 as follows:
    # For a normal distribution approximation:
    # Q1 ~ median - 0.675 * std
    # Q3 ~ median + 0.675 * std
    # Clamp Q1 and Q3 between min and max.
    def approx_quartiles(row, prefix):
        median = row[f"{prefix}_median"]
        std = row[f"{prefix}_std"]
        min_v = row[f"{prefix}_min"]
        max_v = row[f"{prefix}_max"]

        q1 = median - 0.675 * std
        q3 = median + 0.675 * std

        # Clamp quartiles within min and max
        q1 = max(q1, min_v)
        q3 = min(q3, max_v)

        # Edge case: if std=0 or q1/q3 calculation results in values outside min/max,
        # fallback to fixed quartiles (e.g. min, median, max)
        if q1 > median:
            q1 = median
        if q3 < median:
            q3 = median

        return q1, q3

    categories = df['Category'].tolist()

    price_boxes = []
    quantity_boxes = []

    for idx, row in df.iterrows():
        cat = row['Category']

        # Approx quartiles for Price
        price_q1, price_q3 = approx_quartiles(row, "Price")
        price_synth_data = np.array([
            row['Price_min'],
            price_q1,
            row['Price_median'],
            price_q3,
            row['Price_max']
        ])

        price_boxes.append(go.Box(
            y=price_synth_data,
            x=[cat]*len(price_synth_data),
            name=cat,
            boxpoints=False,
            marker_color='rgb(31, 119, 180)',  # blue
            line_color='rgb(31, 119, 180)',
            legendgroup='Price',
            showlegend=(idx == 0),
            hoverinfo='text',
            offsetgroup='Price',
            text=[f"Min: {row['Price_min']}<br>Q1: {price_q1:.2f}<br>Median: {row['Price_median']:.2f}<br>Q3: {price_q3:.2f}<br>Max: {row['Price_max']}"]*len(price_synth_data)
        ))

        # Approx quartiles for Quantity
        quantity_q1, quantity_q3 = approx_quartiles(row, "Quantity")
        quantity_synth_data = np.array([
            row['Quantity_min'],
            quantity_q1,
            row['Quantity_median'],
            quantity_q3,
            row['Quantity_max']
        ])

        quantity_boxes.append(go.Box(
            y=quantity_synth_data,
            x=[cat]*len(quantity_synth_data),
            name=cat,
            boxpoints=False,
            marker_color='rgb(255, 127, 14)',  # orange
            line_color='rgb(255, 127, 14)',
            legendgroup='Quantity',
            showlegend=(idx == 0),
            hoverinfo='text',
            offsetgroup='Quantity',
            text=[f"Min: {row['Quantity_min']}<br>Q1: {quantity_q1:.2f}<br>Median: {row['Quantity_median']:.2f}<br>Q3: {quantity_q3:.2f}<br>Max: {row['Quantity_max']}"]*len(quantity_synth_data)
        ))

    fig = go.Figure()

    for b in price_boxes:
        fig.add_trace(b)
    for b in quantity_boxes:
        fig.add_trace(b)

    fig.update_layout(
        title="Variation in Price and Quantity Sold Across Product Categories",
        xaxis=dict(
            title="Product Category",
            type='category',
            categoryorder='array',
            categoryarray=categories,
        ),
        yaxis=dict(
            title="Value Range",
            zeroline=False,
            autorange=True,
        ),
        boxmode='group',
        legend=dict(
            title="Legend",
            itemsizing='constant',
            traceorder='grouped',
        ),
        margin=dict(t=60, b=60),
        width=900,
        height=600,
    )

    # Adjust hovertemplate to show the detailed stats per box plot
    for trace in fig.data:
        trace.hovertemplate = trace.text[0] + "<extra></extra>"

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict