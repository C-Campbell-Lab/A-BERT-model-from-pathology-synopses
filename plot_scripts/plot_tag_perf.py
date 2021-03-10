import pandas as pd
import numpy as np
from zipfile import ZipFile
import plotly.graph_objects as go


def mk_std_perf(zip_p):
    zip_f = ZipFile(zip_p, "r")
    namelist = zip_f.namelist()

    precisions = []
    recalls = []
    f1s = []

    for fn in namelist:
        if "Perf" in fn:
            with zip_f.open(fn) as csv_r:
                df = pd.read_csv(csv_r, index_col=0)
                precisions.append(df["Precision"].to_list())
                recalls.append(df["Recall"].to_list())
                f1s.append(df["F1 Score"].to_list())
    labels = df["Tag"].to_list()

    f1s_arr = np.array(f1s)
    precisions_arr = np.array(precisions)
    recalls_arr = np.array(recalls)
    performance = pd.DataFrame(
        {
            "Tag": labels,
            "F1 Score": f1s_arr.mean(axis=0),
            "F1 Score_std": f1s_arr.std(axis=0),
            "Precision": precisions_arr.mean(axis=0),
            "Precision_std": precisions_arr.std(axis=0),
            "Recall": recalls_arr.mean(axis=0),
            "Recall_std": recalls_arr.std(axis=0),
        }
    )
    return performance


def plot_tag_perf_with_std(performance, perf_average, perf_err):
    TEMPLATE = "plotly_white"

    y_title = "F1 Score"
    performance.sort_values(
        y_title,
        inplace=True,
    )
    fig = go.Figure()
    x = performance["Tag"]
    marker_symbols = ["square", "x"]
    fig.add_shape(
        type="line",
        x0=0.01,
        y0=perf_average,
        x1=0.99,
        y1=perf_average,
        xref="paper",
        line=dict(color="lightgray", width=2, dash="dash"),
    )
    for idx, measure in enumerate(["Precision", "Recall"]):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=performance[measure],
                error_y=dict(
                    color="lightgray",
                    type="data",
                    array=performance[f"{measure}_std"],
                    visible=False,
                ),
                marker_color="lightgray",
                marker_symbol=marker_symbols[idx],
                marker_size=10,
                mode="markers",
                name=measure,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=performance[y_title],
            error_y=dict(
                color="lightgray",
                type="data",
                array=performance[f"{y_title}_std"],
                visible=True,
            ),
            marker_color="crimson",
            mode="markers+text",
            text=[f"{v:.02f}" for v in performance[y_title]],
            marker_size=10,
            name=y_title,
        )
    )

    fig.update_traces(textposition="middle right")

    x_loc = len(performance) - 2
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
        xaxis_title="Semantic Label",
        yaxis_title="Metrics",
        showlegend=True,
        annotations=[
            dict(
                x=x_loc,
                y=perf_average,
                xref="x",
                yref="y",
                text=f"Micro {y_title}: {perf_average:.03f}±{perf_err:.03f}",
                showarrow=False,
                font=dict(size=15),
            ),
        ],
    )
    return fig
