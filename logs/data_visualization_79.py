# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-17 00:04:46

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # Ensure the 'Year-Month' is string for x-axis labeling
    data = data_raw.copy()
    data['Year-Month'] = data['Year-Month'].astype(str)

    # Aggregate sum of Amount grouped by Year-Month and Category
    agg_df = data.groupby(['Year-Month', 'Category'], as_index=False)['Amount'].sum()

    # Get unique categories and months
    categories = agg_df['Category'].unique()
    year_months = sorted(agg_df['Year-Month'].unique())

    # Create a trace for each category
    traces = []
    for cat in categories:
        df_cat = agg_df[agg_df['Category'] == cat]
        # Align data by all year_months, fill missing with 0
        amounts = pd.Series(index=year_months, dtype=float)
        amounts.loc[df_cat['Year-Month']] = df_cat['Amount'].values
        amounts = amounts.fillna(0)

        traces.append(
            go.Bar(
                name=cat,
                x=year_months,
                y=amounts,
            )
        )

    # Create grouped bar chart
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='group',
        title="Total Amount for each Category across Year-Month",
        xaxis_title="Year-Month",
        yaxis_title="Total Amount",
        legend_title="Category",
        xaxis=dict(tickangle=45)
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict