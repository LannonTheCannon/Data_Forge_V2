# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-28 17:11:46

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio
    from scipy.signal import find_peaks






    # Extract "Total Sales" column
    sales = data_raw["Total Sales"]

    # Calculate central tendency measures
    mean_val = sales.mean()
    median_val = sales.median()

    # Determine bin size automatically using Freedman-Diaconis rule for better distribution revelation
    q75, q25 = np.percentile(sales, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(sales) ** (-1/3))
    if bin_width == 0:
        bin_width = (sales.max() - sales.min()) / 10
    bins = int(np.ceil((sales.max() - sales.min()) / bin_width))
    if bins < 1:
        bins = 10

    # Histogram data (using numpy histogram to get bin counts and edges)
    counts, bin_edges = np.histogram(sales, bins=bins)

    # Find peaks in histogram counts to detect multi-modality
    peaks_idx, _ = find_peaks(counts)
    peaks_x = bin_edges[peaks_idx] + np.diff(bin_edges[:2])/2 if len(peaks_idx)>0 else []

    # Base histogram trace
    hist_trace = go.Histogram(
        x=sales,
        nbinsx=bins,
        marker=dict(color='rgba(0,123,255,0.7)', line=dict(color='rgba(0,0,0,0.8)', width=1)),
        showlegend=False,
        name='Total Sales'
    )

    # Lines for mean and median
    mean_line = go.Scatter(
        x=[mean_val, mean_val],
        y=[0, max(counts)*1.1],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Mean',
        hoverinfo='skip'
    )
    median_line = go.Scatter(
        x=[median_val, median_val],
        y=[0, max(counts)*1.1],
        mode='lines',
        line=dict(color='green', width=2, dash='dot'),
        name='Median',
        hoverinfo='skip'
    )

    # Create figure and add traces
    fig = go.Figure()
    fig.add_trace(hist_trace)
    fig.add_trace(mean_line)
    fig.add_trace(median_line)

    # Annotations for mean and median
    annotations = [
        dict(
            x=mean_val,
            y=max(counts)*1.05,
            xref='x',
            yref='y',
            text=f"Mean: {mean_val:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            font=dict(color='red')
        ),
        dict(
            x=median_val,
            y=max(counts)*1.05,
            xref='x',
            yref='y',
            text=f"Median: {median_val:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-45,
            font=dict(color='green')
        )
    ]

    # If multiple peaks detected, add annotation for multi-modality
    if len(peaks_idx) > 1:
        # Add markers on peaks
        for px in peaks_x:
            fig.add_trace(go.Scatter(
                x=[px],
                y=[counts[peaks_idx[np.where(peaks_x==px)[0][0]]]],
                mode='markers',
                marker=dict(color='orange', size=10, symbol='star'),
                showlegend=False,
                hoverinfo='text',
                text=f'Mode candidate at {px:.2f}'
            ))
        # Add general annotation for multi-modality
        annotations.append(
            dict(
                x=(sales.min() + sales.max())/2,
                y=max(counts)*1.15,
                xref='x',
                yref='y',
                text="Multiple modes detected",
                showarrow=False,
                font=dict(color='orange', size=14, family='Arial Black')
            )
        )

    fig.update_layout(
        title="Distribution of Total Sales with Range, Central Tendency, Skewness, and Multi-modality",
        xaxis_title="Total Sales",
        yaxis_title="Frequency",
        bargap=0.05,
        annotations=annotations,
        margin=dict(t=100),
        hovermode='x unified'
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict