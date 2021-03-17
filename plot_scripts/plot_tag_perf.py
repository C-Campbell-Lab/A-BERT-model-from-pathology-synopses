from zipfile import ZipFile

import pandas as pd
import plotly.graph_objects as go


def mk_std_perf(zip_p):
    zip_f = ZipFile(zip_p, "r")
    namelist = zip_f.namelist()

    precisions = []
    recalls = []
    f1s = []
    labs = []
    labels = []
    for fn in namelist:
        if "Perf" in fn:
            with zip_f.open(fn) as csv_r:
                df = pd.read_csv(csv_r, index_col=0)
                precisions.extend(df["Precision"].to_list())
                recalls.extend(df["Recall"].to_list())
                f1s.extend(df["F1 Score"].to_list())
                labels.extend(df["Tag"].to_list())
                lab = fn.split("/")[1]
                labs.extend(lab for _ in range(len(df)))

    performance = pd.DataFrame(
        {
            "Lab": labs,
            "Label": labels,
            "F1 Score": f1s,
            "Precision": precisions,
            "Recall": recalls,
        }
    )
    aggs = ["mean", "std"]
    perf_agg = performance.groupby("Label").agg(aggs).droplevel(0, axis=1)
    perf_agg.columns = [
        f"{metric}_{agg}"
        for metric in ["F1 Score", "Precision", "Recall"]
        for agg in aggs
    ]
    perf_agg.reset_index(inplace=True)
    return performance, perf_agg


def plot_tag_perf_with_std(performance, perf_average, perf_err):
    TEMPLATE = "plotly_white"

    y_title = "F1 Score"
    performance.sort_values(
        f"{y_title}_mean",
        inplace=True,
    )
    fig = go.Figure()
    x = performance["Label"]
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
                y=performance[f"{measure}_mean"],
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
            y=performance[f"{y_title}_mean"],
            error_y=dict(
                color="lightgray",
                type="data",
                array=performance[f"{y_title}_std"],
                visible=True,
            ),
            marker_color="crimson",
            mode="markers+text",
            text=[f"{v:.02f}" for v in performance[f"{y_title}_mean"]],
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


def tag_perf_latex(dev_df):
    print(
        dev_df.sort_values("F1 Score_mean", ascending=False)
        .groupby(dev_df.columns.str[:3], axis=1)
        .apply(lambda x: round(x, 3).astype(str).apply(" ± ".join, 1))
        .rename(
            {"Lab": "Label", "F1 ": "F1 Score", "Pre": "Precision", "Rec": "Recall"},
            axis=1,
        )
        .loc[:, ["Label", "F1 Score", "Precision", "Recall"]]
        .to_latex(index=False)
    )


if __name__ == "__main__":
    import sys

    df, df_agg = mk_std_perf(sys.argv[1])
    df.to_csv("perf.csv")
    # df_agg.droplevel(0, axis=1).to_csv("perf_agg.csv")
    tag_perf_latex(df_agg)
    df_agg.to_csv("perf_agg.csv")
