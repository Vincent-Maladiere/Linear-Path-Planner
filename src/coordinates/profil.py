import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_profil(filename):

    df = pd.read_csv(filename)

    col_speeds = ["speed_1_a", "speed_1_b", "speed_2_a", "speed_2_b"]
    fig_speed = make_subplots(rows=2, cols=2, shared_yaxes=True)
    for idx, col in enumerate(col_speeds):
        fig_speed.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
            ),
            row=idx // 2 + 1,
            col=idx % 2 + 1,
        )
    fig_speed.update_layout(dict(title="Speed (cm/s)"))
    fig_speed.show()

    fig_area = make_subplots(rows=1, cols=2, shared_yaxes=True)
    for idx, col in enumerate(["area_left", "area_right"], 1):
        fig_area.add_trace(
            go.Scatter(
                x=df.iloc[1:].index,
                y=df.iloc[1:][col],
                name=col,
            ),
            row=1,
            col=idx
        )
    fig_area.update_layout(dict(title="Area diff (m2)"))
    fig_area.show()

    col_angles = ["theta_1_a", "theta_1_b", "theta_2_a", "theta_2_b"]
    fig_angle = make_subplots(rows=2, cols=2, shared_yaxes=False)
    for idx, col in enumerate(col_angles):
        fig_angle.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
            ),
            row=idx // 2 + 1,
            col=idx % 2 + 1,
        )
    fig_angle.update_layout(dict(title="Theta (degree)"))
    fig_angle.show()
