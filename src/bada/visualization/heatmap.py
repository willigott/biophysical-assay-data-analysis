import numpy as np
import plotly.graph_objects as go


def plot_plate_data(
    plate_data: np.ndarray,
    cols: list[str],
    rows: list[str],
    title: str,
    color_scale: str = "RdBu_r",
    colorbar_title: str = "",
) -> go.Figure:
    heatmap_fig = go.Figure()
    # fmt: off
    heatmap_fig.add_trace(
        go.Heatmap(
            z=plate_data,
            x=cols,
            y=rows,
            colorscale=color_scale,
            text=[[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in plate_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    side="right"
                ),
                thickness=20,
                len=0.8
            ),
            xgap=2,
            ygap=2,
        )
    )
    # fmt: on
    heatmap_fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=400 if len(rows) <= 8 else 600,
        width=None,
        yaxis_autorange="reversed",
        margin=dict(t=60, r=80, b=60, l=60),
        xaxis=dict(
            side="top",
            tickmode="linear",
            dtick=1,
            tickangle=0,
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=1,
            griddash="solid",
            tickson="boundaries",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=1,
            griddash="solid",
            tickson="boundaries",
        ),
        plot_bgcolor="white",
    )

    return heatmap_fig
