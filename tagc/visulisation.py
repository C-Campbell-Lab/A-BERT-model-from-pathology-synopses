from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.preprocessing import normalize

TEMPLATE = "plotly_white"


def plot_tag_stat(tag_df: pd.DataFrame):
    s = tag_df.groupby("tag")["count"].sum()
    sort_idx = []
    for i in s.sort_values(ascending=False).index:
        sort_idx.append(i + "train")
        sort_idx.append(i + "test")
    tag_df["idx"] = tag_df["tag"] + tag_df["for"]
    tag_df = tag_df.set_index("idx")
    sort_idx = [idx for idx in sort_idx if idx in tag_df.index]
    tag_df = tag_df.loc[sort_idx]
    fig = px.bar(
        tag_df,
        x="tag",
        y="count",
        color="for",
        title="Label Summary",
        text="count",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
    )
    return fig


def state_plot(df: pd.DataFrame, thresh=15):

    markers = df.tag.value_counts()
    tag_value_counts = markers >= thresh
    show_legends = set(tag_value_counts.index.to_numpy()[tag_value_counts])
    df["show_legends"] = df["tag"].map(lambda x: x in show_legends)

    raw_symbols = SymbolValidator().values
    colors = px.colors.qualitative.Plotly
    symbol_dict = {}
    color_dict = {}
    color_len = len(colors)
    for idx, tag in enumerate(markers.index):
        symbol_idx = idx // color_len
        color_idx = idx % color_len
        symbol_dict[tag] = raw_symbols[symbol_idx]
        color_dict[tag] = colors[color_idx]
    df["color"] = df.tag.map(color_dict)
    df["symbol"] = df.tag.map(symbol_dict)

    fig = go.Figure()
    sel_tags = sorted(show_legends, key=len)
    for sel_tag in sel_tags:
        tmp_df = df.loc[df.tag == sel_tag, :]
        fig.add_trace(
            go.Scatter(
                x=tmp_df["D1"],
                y=tmp_df["D2"],
                mode="markers",
                marker_color=tmp_df["color"],
                marker_symbol=tmp_df["symbol"],
                hovertemplate="<b>tag:%{text}</b><br>pred_tag:%{customdata}",
                text=tmp_df["tag"].to_list(),
                customdata=tmp_df["pred_tag"].to_list(),
                showlegend=True,
                name=sel_tag,
            )
        )

    no_legend_df = df.loc[~df["show_legends"], :]
    fig.add_trace(
        go.Scatter(
            x=no_legend_df["D1"],
            y=no_legend_df["D2"],
            mode="markers",
            opacity=0.5,
            marker_color=no_legend_df["color"],
            marker_symbol=no_legend_df["symbol"],
            showlegend=False,
            hovertemplate="<b>tag:%{text}</b><br>pred_tag:%{customdata}",
            text=no_legend_df["tag"].to_list(),
            customdata=no_legend_df["pred_tag"].to_list(),
            name="",
        )
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        legend=dict(
            orientation="h",
        ),
        width=1280,
        height=600,
    )
    return fig


def plot_judge_num(j_tag_num, mode="Correct"):
    data = defaultdict(list)
    for k, v in j_tag_num.items():
        counter = Counter(v)
        data["Correct Num"].extend(counter.keys())
        data["Count"].extend(counter.values())
        data["Tag Num"].extend(k for _ in range(len(counter)))

    tmp_df = (
        pd.pivot_table(
            pd.DataFrame(data), index=["Tag Num", "Correct Num"], values="Count"
        )
        .unstack()
        .fillna(0)
    )
    tmp_df.columns = [f"{c[1]} {mode}" for c in tmp_df.columns]

    judge_df_rate = tmp_df.copy()
    j_df_sum = judge_df_rate.sum(axis=1)

    for idx, row in enumerate(judge_df_rate.iterrows()):
        judge_df_rate.values[idx] = row[1] / j_df_sum.iloc[idx]

    data = []
    judge_df = tmp_df.reset_index()
    tag_num = judge_df["Tag Num"]
    for col in judge_df.columns[1:]:
        y = judge_df[col]
        text = [f"{v * 100:.02f} %" for v in judge_df_rate[col]]
        data.append(
            go.Bar(
                name=col,
                x=tag_num,
                y=y,
                text=text,
                textposition="auto",
            ),
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        barmode="stack",
        xaxis_title="Tag Number",
        yaxis_title="Count",
        legend_title=f"{mode} Predition",
    )
    return fig


def kw_plot(top_key):
    top_n = 5
    labels = []
    values = []
    parents = []
    for idx, (k, v) in enumerate(top_key.items()):
        labels.extend(i[0] for i in v[:top_n])
        tmp_v = [i[1] for i in v]
        tmp_v = normalize([tmp_v])[0]
        values.extend(round(v, 3) for v in tmp_v[:top_n])
        parents.extend(k for _ in range(top_n))
    kws_df = pd.DataFrame({"influence": values, "reason": labels, "tag": parents})
    heat_df = kws_df.groupby("tag").agg(list)

    vs = heat_df["influence"].values
    mat = np.zeros(
        (len(vs), len(vs[0])),
    )
    for i, r in enumerate(vs):
        for j, v in enumerate(r):
            mat[i][j] = v

    sort_ = mat[:, 0].argsort()
    z = heat_df["influence"].values[sort_].tolist()
    y = heat_df.index[sort_].to_list()
    z_text = heat_df["reason"].values[sort_].tolist()
    fig = ff.create_annotated_heatmap(
        z,
        y=y,
        x=list(range(1, 1 + top_n, 1)),
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
    )
    return fig


def plot_summary(data):
    tree_data = defaultdict(dict)
    for n in range(1, 7):
        labels = []
        parents = []
        values = []
        for type_ in ("less", "more", "equal"):
            if type_ != "equal":
                count = 0
                for k_, v_ in data[type_][n].items():
                    tmp_v = []
                    for k, v in v_.items():
                        labels.append(f"{k} correct")
                        tmp_v.append(v)
                        parents.append(f"pred {k_ + n}")

                    labels.append(f"pred {k_ + n}")
                    sum_v = sum(tmp_v)
                    values.extend(tmp_v)
                    values.append(sum_v)
                    count += sum_v
                    parents.append(type_)
                labels.append(type_)
                values.append(count)
                parents.append("")
            else:
                tmp_v = []
                for k, v in data[type_][n].items():
                    labels.append(f"{k} correct")
                    tmp_v.append(v)
                    parents.append(type_)

                values.extend(tmp_v)

                values.append(sum(tmp_v))
                labels.append(type_)
                parents.append("")

        tree_data[n]["labels"] = labels
        tree_data[n]["parents"] = parents
        tree_data[n]["values"] = values

    col_num = 3
    row_num = int(6 / col_num)
    fig = make_subplots(
        cols=col_num,
        rows=row_num,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
        specs=[[{"type": "domain"} for _ in range(col_num)] for _ in range(row_num)],
        subplot_titles=[f"{tag_num} Tag" for tag_num in tree_data.keys()],
    )

    for idx, tmp_tree in enumerate(tree_data.values()):

        col = idx % col_num + 1
        row = idx // col_num + 1
        fig.add_trace(
            go.Sunburst(
                labels=tmp_tree["labels"],
                parents=tmp_tree["parents"],
                values=tmp_tree["values"],
                branchvalues="total",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
    )
    return fig


def plot_tag_performance(performance: pd.DataFrame, overall, auc=False):
    y_title = "AUC" if auc else "F1 Score"
    performance.sort_values(
        y_title,
        inplace=True,
    )
    fig = go.Figure()
    x = performance["Tag"]
    marker_symbols = ["square", "cross"]
    perf_average = overall[y_title]
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
                text=f"Micro {y_title}: {perf_average:.03f}",
                showarrow=False,
                font=dict(size=15),
            ),
        ],
    )
    return fig


def plot_num_performance(performance_n: pd.DataFrame):
    fig = go.Figure()
    x = performance_n.index
    size = performance_n["Count"]
    for c in ["F1 Score"]:
        y = performance_n[c]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=c,
                mode="markers+text",
                marker=dict(
                    size=size,
                    sizemode="area",
                    sizeref=2.0 * max(size) / (40.0 ** 2),
                    sizemin=5,
                ),
                text=[f"{v:.03f}" for v in y],
                textposition="middle right",
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )
    fig.update_xaxes(title_text="Tag Number")

    return fig
