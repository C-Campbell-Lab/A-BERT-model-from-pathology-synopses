import json
import os
import pickle
import re
from typing import List

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.preprocessing import MultiLabelBinarizer

from .domain import Mask
from .io_utils import load_datazip, load_json, prepare_dataset, prepare_model
from .mask_explain import MaskExplainer, plot_explanation
from .model import StandaloneModel
from .validation import dimension_reduction_plot, get_tag_states


def load_state(state_p: str):
    with open(state_p, "rb") as plk:
        state = pickle.load(plk)
    return state


def dump_state(states, state_p="state"):
    with open(state_p, "wb") as plk:
        pickle.dump(states, plk)
    return state_p


def empty_bar():
    return px.bar(x=["None"], y=[0])


class Server:
    def __init__(self, state=None):
        self.dataset_p = "dataset.zip"
        self.model_p = "model"
        self.unlabelled_p = "data/unlabelled.json"
        self.state = state

        self.init_static()
        self.init_state()
        self.init_plot()

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def init_static(self):
        if not os.path.exists(self.dataset_p):
            prepare_dataset(self.dataset_p)
        if not os.path.exists(self.model_p):
            prepare_model(self.model_p)

    def init_state(self):
        self.rawdata = load_datazip(self.dataset_p)
        self.unlabelled = load_json(self.unlabelled_p)

        self.model = StandaloneModel.from_path(self.model_p)
        self.mlb = MultiLabelBinarizer().fit(self.rawdata.y_tags)
        self.mask_explainer = MaskExplainer(self.mlb)
        if self.state is None:
            self.state = get_tag_states(self.model, self.rawdata, self.mlb)

    def init_plot(self):
        self.fig, self.dimension_reducer = dimension_reduction_plot(
            self.state, method_n="tsne", n_components=2, dash=True
        )

    def plot(self):
        app = self.app
        input_id = "input"
        dot_id = "dot-interactions"
        case_id = "cases"
        mask_id = "mask"
        app.layout = html.Div(
            [
                html_input(id_=input_id),
                html.H2("t-SNE"),
                html_dots(dot_id, self.fig),
                html.H2("Explanation"),
                html_case(id_=case_id),
                html_mask(id_=mask_id),
            ]
        )

        @app.callback(
            [Output(case_id, "children"), Output(mask_id, "figure")],
            [
                Input(component_id=input_id, component_property="value"),
                Input(dot_id, "clickData"),
            ],
        )
        def update_output_div(input_value, clickData):
            if input_value is not None:
                return self._case_plot({"COMMENT": input_value})
            elif clickData is not None:
                return self._display_click_data(clickData)
            return [[], empty_bar()]

    def _display_click_data(self, clickData):

        customdata = clickData["points"][0]["customdata"]
        idx = customdata[0]
        from_ = customdata[1]
        if from_ == "unlabelled":
            case = self.unlabelled[idx]
        else:
            data = self.rawdata.retrive(from_, idx)
            case = data["text"]
        return self._case_plot(case)

    def _case_plot(self, case):
        rets = self.mask_explainer.explain(self.model, case)
        childrend = []
        for ret in rets:
            importance = ret.importance
            pos_key_marks = [p[0] for p in importance][:5]
            childrend.append(html.H3(ret.tag))
            childrend.append(draw_color(case, pos_key_marks))

        fig = plot_explanation(rets, dash=True)
        return [childrend, fig]


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


def draw_color(case, key_masks: List[Mask]):
    finder = re.compile(r"[\w<>]+")
    children = []
    pos_mark = "<POS>"
    pos_style = {"color": "red"}
    for mask in key_masks:
        case = mask.mark(case, pos_mark)

    text = json.dumps(case, indent=2)
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
