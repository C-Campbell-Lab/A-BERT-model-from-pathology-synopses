import json

import pandas as pd
import plotly.graph_objects as go


def ac_plot_pipe(collect_ac_p, dst="."):
    ac_history = load_json(collect_ac_p)
    ac_up_df, rd_up_df = mk_ac_df(ac_history, return_df=False)
    fig = plot_ac_compare(ac_up_df, rd_up_df)
    fig.write_image(f"{dst}/acl_result.pdf")


def load_json(path):
    with open(path, "r", encoding="utf-8") as js_:
        return json.load(js_)


def mk_ac_df(ac_history, return_df=False):
    sizes = []
    is_active = []
    f1_scores = []
    labs = []
    for k, v in ac_history.items():
        parts = k.split("/")
        size = int(parts[-1].split("_")[0])
        active = False if "R" in parts[1] else True
        lab = parts[1].replace("R", "")
        f1_score = v[0]["F1 Score"]
        sizes.append(size)
        is_active.append(active)
        labs.append(lab)
        f1_scores.append(f1_score)
    df = pd.DataFrame(
        dict(sizes=sizes, is_active=is_active, labs=labs, f1_scores=f1_scores)
    )
    ac_up_df = df_cal(df, True)
    rd_up_df = df_cal(df, False)
    if return_df:
        return ac_up_df, rd_up_df, df
    else:
        return ac_up_df, rd_up_df


def df_cal(df, is_active):
    out = (
        df.loc[df["is_active"] == is_active]
        .groupby(["sizes"])["f1_scores"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return out


def plot_ac_compare(ac_up_df, rd_up_df):
    TEMPLATE = "plotly_white"
    fig = go.Figure()
    colors = ["gray", "cornflowerblue"]
    p1 = ac_up_df.iloc[:6, :]
    p2 = ac_up_df.iloc[5:, :]
    for idx, df in enumerate([p1, p2]):
        x = df["sizes"]
        y = df["mean"]
        e = df["std"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=dict(
                    color="lightgray",
                    type="data",
                    array=e,
                    visible=True,
                ),
                mode="lines+markers+text",
                text=[f"{v:.03f}" for v in y],
                marker_color=colors[idx],
                line=dict(color=colors[idx], width=2),
                textposition="bottom right",
                name="active learning",
                showlegend=idx == 0,
            ),
        )

    x = rd_up_df["sizes"]
    y = rd_up_df["mean"]
    e = rd_up_df["std"]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                color="lightgray",
                type="data",
                array=e,
                visible=True,
            ),
            mode="lines+markers+text",
            text=[f"{v:.03f}" for v in y],
            marker_color="gray",
            line=dict(color="gray", width=2, dash="dash"),
            textposition="bottom right",
            name="random sampling",
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1000,
        height=600,
        xaxis_title="Training Size",
        yaxis_title="Micro F1",
        showlegend=True,
        legend=dict(
            orientation="h",
        ),
    )
    rd_max = rd_up_df["mean"].max()
    ac_max = ac_up_df["mean"].max()
    fig.add_shape(
        type="line",
        x0=0.01,
        y0=rd_max,
        x1=0.99,
        y1=rd_max,
        xref="paper",
        line=dict(color="lightgray", width=2, dash="dash"),
    )
    fig.add_annotation(x=100, y=rd_max, text="F1 (400 Random samples)", showarrow=False)
    fig.add_annotation(x=350, y=0.8, text="Plateau", showarrow=False)
    fig.add_annotation(
        x=450, y=ac_max, align="left", text="<b>Active learning</b>", showarrow=False
    )
    fig.add_annotation(
        x=450, y=rd_max, align="left", text="<b>Random sampling</b>", showarrow=False
    )
    fig.add_annotation(
        x=0,
        y=-0.12,
        xref="paper",
        yref="paper",
        align="left",
        text="n=4 independent experiments",
        showarrow=False,
    )

    return fig
