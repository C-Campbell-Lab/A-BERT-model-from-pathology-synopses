import json

import re
import pandas as pd
import plotly.graph_objects as go


def load_json(path):
    with open(path, "r") as js_:
        return json.load(js_)


def mk_fd_df(feedback: dict):
    iter_parser = re.compile(r"\d")
    experts = []
    validation = []
    scores = []
    iterations = []
    labs = []
    fnames = []
    for k, v in feedback.items():
        if "after" in k:
            continue
        parts = k.split("/")
        experts.append(parts[2])
        is_val = True if "overall" in k else False
        validation.append(is_val)
        lab = parts[1]
        fname = parts[-1]
        iteration = (
            iter_parser.findall(fname)[-1]
            if (not is_val or lab == "lab0")
            else fname.split("_")[0]
        )
        iterations.append(int(iteration))
        fnames.append(fname)
        labs.append(lab)
        record = list(v[0].values())[0]
        if isinstance(record, dict):
            scores.append(record["f1"])
        else:
            scores.append(v[0]["F1 Score"])
    return pd.DataFrame(
        dict(
            experts=experts,
            iterations=iterations,
            labs=labs,
            fnames=fnames,
            validation=validation,
            scores=scores,
        )
    )


def plot_feedback(start_f1, start_err, fd_df):
    TEMPLATE = "plotly_white"

    fig = go.Figure()
    experts = ["feedbackC", "feedbackM"]
    validation = [True, False]
    marker_color = ["navy", "crimson"]

    phase = []
    f1 = []
    err = []
    sources = []
    for val in validation:
        for exp_idx, exp in enumerate(experts):
            df = fd_df.loc[(fd_df["experts"] == exp) & (fd_df["validation"] == val)]
            plot_df = df.groupby(["iterations"])["scores"].agg(["mean", "std"])
            if val:
                y = [start_f1] + plot_df["mean"].to_list()
                x = _mk_x(y)
                e = [start_err] + plot_df["std"].to_list()
                source = f"Val{exp_idx}"
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        error_y=dict(
                            color="lightgray",
                            type="data",
                            array=e,
                            visible=False,
                        ),
                        mode="lines",
                        marker_color=marker_color[exp_idx],
                        line=dict(color=marker_color[exp_idx], width=2),
                        textposition="bottom right",
                        name=source,
                    )
                )

            else:
                y = plot_df["mean"].to_list()
                x = _mk_x(y)
                e = plot_df["std"].to_list()
                source = f"Pathologist{exp_idx}"
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        error_y=dict(
                            color="lightgray",
                            type="data",
                            array=e,
                            visible=False,
                        ),
                        mode="markers+text",
                        text=[f"{v:.03f}" for v in y],
                        marker_color=marker_color[exp_idx],
                        textposition="middle right",
                        name=source,
                    )
                )
            phase.extend(x)
            f1.extend(y)
            err.extend(e)
            sources.extend(source for _ in x)

    fig.update_layout(
        template=TEMPLATE,
        width=960,
        height=500,
        xaxis_title="Data consumed by model",
        yaxis_title="Micro F1",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
    )
    out_df = pd.DataFrame(dict(phase=phase, f1=f1, err=err, sources=sources))
    return fig, out_df


def _mk_x(y):
    return [
        f"Dev{'+' if add > 0 else ''}{add * 100 if add > 0 else ''}"
        for add in range(len(y))
    ]
