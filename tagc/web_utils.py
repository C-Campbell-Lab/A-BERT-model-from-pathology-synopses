import json
import re
from typing import List

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

from .domain import Mask


def html_input(id_):
    return html.Div(["Input: ", dcc.Input(id=id_, type="text")])


def html_dots(id_, fig):
    return dcc.Graph(id=id_, figure=fig)


def html_case(id_="click-data"):
    return html.Div(
        id=id_,
        style={
            "white-space": "pre",
            "overflowX": "scroll",
        },
    )


def html_mask(id_="mask"):
    return dcc.Graph(id=id_)


def dict_to_str(case):
    return json.dumps(case, indent=2)


def draw_color(case, key_masks: List[Mask]):
    finder = re.compile(r"[\w<>]+")
    children = []
    pos_mark = "<POS>"
    pos_style = {"color": "red"}
    for mask in key_masks:
        case = mask.mark(case, pos_mark)

    text = dict_to_str(case)
    cur = 0
    for part in finder.finditer(text):
        g = part.group(0)
        start, end = part.span()
        if pos_mark in g:
            children.append(text[cur:start])
            children.append(html.Span(f'{g.replace(pos_mark, "")} ', style=pos_style))
        else:
            children.append(text[cur:end])
        cur = end
    children.append(text[end:])
    return html.P(children)


def html_checkbox(id_):
    # options = [{"label": t, "value": t} for t in TAG]
    return dcc.Checklist(id=id_)


def html_submit(id_):
    return html.Button("Submit", id=id_)


def empty_bar():
    return px.bar(x=["None"], y=[0])
