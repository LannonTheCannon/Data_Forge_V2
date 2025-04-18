# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-16 20:59:12

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.express as px
    import plotly.io as pio





    fig = px.bar(
        data_raw,
        x="Offender_Race",
        y="Count",
        color="Offender_Gender",
        barmode="stack",
        title="Count of Offenders by Race Stacked by Gender",
        labels={
            "Offender_Race": "Offender_Race",
            "Count": "Count",
            "Offender_Gender": "Offender_Gender"
        }
    )

    fig.update_layout(
        xaxis_title="Offender_Race",
        yaxis_title="Count"
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict