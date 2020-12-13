import json
from pathlib import Path
from statistics import mean, stdev

import plotly.graph_objects as go


def load_json(path):
    with open(path, "r") as js_:
        return json.load(js_)


output = Path("/mnt/e/outputsS/")
active_f = output.glob("eval_over*")
cath_p = output / "Cath"
ss = list(map(load_json, cath_p.glob("eval_over*")))
cc = list(map(load_json, cath_p.glob("*overall*")))
mm_p = output / "Mona"
mm = list(map(load_json, mm_p.glob("eval_over*")))
mmc = list(map(load_json, mm_p.glob("*overall*")))


TEMPLATE = "plotly_white"

fig = go.Figure()

yc = [i["F1 Score"] for i in cc]
xc = ["Dev"] + [f"Dev+{add}" for add in range(100, 1000, 100)][: len(yc)]
fig.add_trace(
    go.Scatter(
        x=xc,
        y=yc,
        mode="lines",
        marker_color="crimson",
        line=dict(color="green", width=2),
        textposition="bottom right",
        name="Val1",
    )
)

yc = [i["F1 Score"] for i in mmc]
xc = xc
fig.add_trace(
    go.Scatter(
        x=xc,
        y=yc,
        mode="lines",
        marker_color="crimson",
        line=dict(color="crimson", width=2),
        textposition="bottom right",
        name="Val2",
    )
)


y = [i["Cathy"]["f1"] for i in ss]
x = xc[: len(y)]
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=[f"{v:.02f}" for v in y],
        marker_color="green",
        textposition="middle right",
        name="Pathologist1",
    )
)

y = [i["Monalisa"]["f1"] for i in mm]
x = xc[: len(y)]
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=[f"{v:.02f}" for v in y],
        marker_color="crimson",
        textposition="middle right",
        name="Pathologist2",
    )
)


fig.update_layout(
    template=TEMPLATE,
    width=800,
    height=600,
    xaxis_title="Evaluation Batch",
    yaxis_title="Micro F1",
    showlegend=True,
    legend=dict(
        orientation="h",
    ),
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)

exp = 0.783


all_f1 = [i["Cathy"]["f1"] for i in ss] + [i["Monalisa"]["f1"] for i in mm]

mean_all = mean(all_f1)
std_all = stdev(all_f1)


print(f"{mean_all:.03f}±{std_all:.03f}")
fig.write_image(str(output / "acL2.pdf"))
