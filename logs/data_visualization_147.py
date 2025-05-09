# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-27 17:01:19

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # Pivot data to have job_title as rows, employee_residence as columns, average_salary_in_usd as values
    heatmap_data = data_raw.pivot(index='job_title', columns='employee_residence', values='average_salary_in_usd')

    # Sort the axes for consistent display (optional, but improves readability)
    heatmap_data = heatmap_data.reindex(index=sorted(heatmap_data.index), 
                                        columns=sorted(heatmap_data.columns))

    # Prepare text for hover tooltips with job_title, employee_residence, and salary
    hover_text = []
    for job in heatmap_data.index:
        hover_text.append([
            f"Job Title: {job}<br>Employee Residence: {res}<br>Average Salary (USD): {heatmap_data.loc[job, res]:,.2f}"
            if not pd.isna(heatmap_data.loc[job, res]) else 
            f"Job Title: {job}<br>Employee Residence: {res}<br>Average Salary (USD): N/A"
            for res in heatmap_data.columns
        ])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title="Salary (USD)"),
        hoverinfo='text',
        text=hover_text,
        zmin=np.nanmin(heatmap_data.values),
        zmax=np.nanmax(heatmap_data.values),
    ))

    fig.update_layout(
        title="Heatmap of Average Salary (USD) by Job Title and Employee Residence",
        xaxis=dict(
            title="Employee Residence (Country Code)",
            tickangle=45,
            tickmode='array',
            tickvals=heatmap_data.columns,
            ticktext=heatmap_data.columns,
            showgrid=False,
            automargin=True
        ),
        yaxis=dict(
            title="Job Title",
            autorange='reversed',  # To have the first job title on top
            tickmode='array',
            tickvals=heatmap_data.index,
            ticktext=heatmap_data.index,
            showgrid=False,
            automargin=True
        ),
        margin=dict(l=120, r=40, t=80, b=160),
        height=600,
        width=1200,
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict