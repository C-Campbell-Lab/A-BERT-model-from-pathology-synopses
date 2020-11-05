from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator

import plotly.figure_factory as ff


def plot_tag_stat(tag_df: pd.DataFrame):
    fig = px.bar(
        tag_df, x="tag", y="count", color="for", title="Tag Stat", text="count"
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
        )
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
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
        top = v[:top_n]
        labels.extend(i[0] for i in top)
        tmp_v = [i[1] for i in top]
        values.extend(round(v, 3) for v in tmp_v)
        parents.extend(k for _ in range(top_n))
    kws_df = pd.DataFrame({"influence": values, "reason": labels, "tag": parents})
    heat_df = kws_df.groupby("tag").agg(list)
    z = heat_df["influence"].values.tolist()
    y = heat_df.index.to_list()
    z_text = heat_df["reason"].values.tolist()
    fig = ff.create_annotated_heatmap(
        z,
        y=y,
        x=list(range(1, 1 + len(z), 1)),
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
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
        horizontal_spacing=0.01,
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
        width=1280,
        height=600,
    )
    return fig


def plot_tag_performance(performance: pd.DataFrame, overall):
    performance = performance.loc[performance["Testing Size"] > 3]
    fig = go.Figure()
    fig = px.scatter(
        performance,
        x="Tag",
        y="F1 Score",
        size="Training Size",
        color="Testing Size",
        text=[f"{v:.02f}" for v in performance["F1 Score"]],
    )
    fig.update_traces(textposition="middle right")
    f1 = overall["f1"]
    x_loc = len(performance) - 2
    fig.update_layout(
        width=1280,
        height=600,
        showlegend=False,
        annotations=[
            dict(
                x=x_loc,
                y=f1,
                xref="x",
                yref="y",
                text=f"F1 Score: {f1:.03f}",
                showarrow=False,
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
        width=1280, height=600, uniformtext_minsize=11, uniformtext_mode="show"
    )
    fig.update_xaxes(title_text="Tag Number")

    return fig
