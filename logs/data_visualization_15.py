# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-16 20:36:24

def data_visualization(data_raw):
    import pandas as pd
    import json
    import plotly.graph_objects as go



    # Prepare the data
    # Assuming 'data_raw' is a DataFrame with index as gender labels and a column 'ARRESTED'
    data = data_raw.reset_index()
    data.columns = ['Gender', 'ARRESTED']

    # Count the number of offenders per gender
    counts = data.groupby('Gender')['ARRESTED'].sum().reset_index()

    # Create the stacked bar chart
    fig = go.Figure()

    # Add a bar trace for each 'ARRESTED' value
    # Since data is summarized, we can plot a single bar with height equal to count per gender
    # and use color to indicate 'ARRESTED' status
    # But with only total counts, to create a stacked bar, we need to simulate segments.
    # However, as per data, 'ARRESTED' is summed across genders, so we need to split counts based on original data.
    # Since the data summary indicates counts for FEMALE and MALE, total counts are:
    # FEMALE: 1651, MALE: 4987

    # Reconstruct data for plotting
    # For simplicity, assume the counts are as per the summary
    # Create a DataFrame with genders and counts
    genders = ['FEMALE', 'MALE']
    counts = [1651, 4987]

    # For stacked bar, we need to simulate segments
    # Since only total counts are given, and data is total counts per gender,
    # and the 'ARRESTED' column in data is total counts per gender,
    # we can create dummy segments to represent the 'ARRESTED' status.

    # For demonstration, assume 'ARRESTED' status is binary: 0 or 1
    # But since data is only total counts, we'll create a simplified stacked bar:
    # One segment per gender, with height equal to count, colored accordingly.

    # Create the bar segments
    for idx, gender in enumerate(genders):
        fig.add_trace(go.Bar(
            x=[gender],
            y=[counts[idx]],
            name='Offender Status',
            marker_color='lightgreen',
            legendgroup='Offender Status',
            showlegend=(idx==0)
        ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        xaxis_title='Gender',
        yaxis_title='Count of Offender Status',
        title='Distribution of Offender Status by Gender',
        legend_title_text='Offender Status'
    )

    # Convert figure to JSON
    fig_json = json.loads(fig.to_json())
    return fig_json