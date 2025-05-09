# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-28 12:03:12

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import qualitative





    # Copy data to avoid modifying original
    df = data_raw.copy()

    # Filter to only months 2, 3, 4 (Feb, Mar, Apr)
    df = df[df['Month'].isin([2, 3, 4])]

    # Create a month label for x-axis
    month_map = {2: "Feb", 3: "Mar", 4: "Apr"}
    df['Month_Label'] = df['Month'].map(month_map)

    # Sort data by Customer Location, Payment Method, then Month for consistent plotting order
    df = df.sort_values(by=['Customer Location', 'Payment Method', 'Month'])

    # We will create a multi-line chart with lines for each (Customer Location, Payment Method) pair.
    # Differentiate lines by color for Payment Method and line dash style for Customer Location.
    # Since there are 10 Customer Locations and 4 Payment Methods, use dash styles for locations and colors for payment methods.
    payment_methods = df['Payment Method'].unique()
    customer_locations = df['Customer Location'].unique()

    # Assign colors for payment methods - use Plotly qualitative palette

    color_palette = qualitative.Plotly
    # Map each payment method to a color cyclically if needed
    color_map = {pm: color_palette[i % len(color_palette)] for i, pm in enumerate(payment_methods)}

    # Assign dash styles to customer locations (there are 10, so pick 10 dash styles)
    dash_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot", "solid", "dot", "dash", "dashdot"]
    dash_map = {loc: dash_styles[i % len(dash_styles)] for i, loc in enumerate(customer_locations)}

    # Create figure
    fig = go.Figure()

    # Add a line trace for each combination of Customer Location and Payment Method
    for loc in customer_locations:
        df_loc = df[df['Customer Location'] == loc]
        for pm in payment_methods:
            df_sub = df_loc[df_loc['Payment Method'] == pm]
            # Ensure data is sorted by Month for line plotting
            df_sub = df_sub.sort_values('Month')
            # Plot line with markers
            fig.add_trace(go.Scatter(
                x=df_sub['Month_Label'],
                y=df_sub['Usage Count'],
                mode='lines+markers',
                name=f"{pm} - {loc}",
                line=dict(color=color_map[pm], dash=dash_map[loc]),
                marker=dict(size=6),
                legendgroup=pm,
                hovertemplate=(
                    "<b>Customer Location:</b> " + loc + "<br>" +
                    "<b>Payment Method:</b> " + pm + "<br>" +
                    "<b>Month:</b> %{x} 2025<br>" +
                    "<b>Usage Count:</b> %{y}<extra></extra>"
                )
            ))

    # Update layout for title, axes, legend
    fig.update_layout(
        title="Monthly Changes in Payment Method Usage by Customer Location (Feb-Apr 2025)",
        xaxis_title="Month (2025)",
        yaxis_title="Usage Count",
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=["Feb", "Mar", "Apr"],
            tickmode='array',
            tickvals=["Feb", "Mar", "Apr"]
        ),
        legend_title="Payment Method - Customer Location",
        legend=dict(
            # Group legend items by Payment Method using legendgroup and tracegroupgap
            tracegroupgap=100,
            itemsizing='constant',
            title_font_size=12,
            font=dict(size=10),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bordercolor="LightGray",
            borderwidth=1,
            bgcolor="White"
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode="x unified"
    )

    # To help user focus on Payment Methods in legend, create custom legend titles by Payment Method:
    # Because each trace name contains both, the legend will show all lines. This is expected for toggling.
    # User can toggle lines by clicking legend entries.

    # Convert to dict and return
    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict