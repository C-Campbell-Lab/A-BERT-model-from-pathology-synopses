import json
import os
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.preprocessing import MultiLabelBinarizer

from .domain import MaskRet
from .io_utils import load_datazip, prepare_dataset, prepare_model
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
        self.state = state

        self.init_static()
        self.init_state()

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def init_static(self):
        if not os.path.exists(self.dataset_p):
            prepare_dataset(self.dataset_p)
        if not os.path.exists(self.model_p):
            prepare_model(self.model_p)

    def init_state(self):
        self.rawdata = load_datazip(self.dataset_p)

        self.model = StandaloneModel.from_path(self.model_p)
        self.mlb = MultiLabelBinarizer().fit(self.rawdata.y_tags)
        self.mask_explainer = MaskExplainer(self.mlb)
        if self.state is None:
            self.state = get_tag_states(self.model, self.rawdata, self.mlb)

    def plot(self):

        app = self.app
        dot_id = "dot-interactions"
        case_id = "cases"
        mask_id = "mask"
        app.layout = html.Div(
            [
                html.H2("t-SNE"),
                html_dots(self.state, id_=dot_id),
                html.H2("Explanation"),
                html_case(id_=case_id),
                html_mask(id_=mask_id),
            ]
        )

        @app.callback(
            [Output(case_id, "children"), Output(mask_id, "figure")],
            [Input(dot_id, "clickData")],
        )
        def display_click_data(clickData):
            if clickData is not None:
                customdata = clickData["points"][0]["customdata"]
                idx = customdata[0]
                from_ = customdata[1]
                data = self.rawdata.retrive(from_, idx)
                case = data["text"]
                rets = self.mask_explainer.explain(self.model, case, for_color=True)
                childrend = []
                fig_rets = []
                for ret in rets:
                    importance = ret.importance
                    pos_kw = [p[0] for p in importance][:5]
                    childrend.append(html.H3(ret.tag))
                    childrend.append(draw_color(case, pos_kw))

                    fig_importance = [(p[0][0], p[1]) for p in importance]
                    fig_rets.append(MaskRet(ret.tag, fig_importance))

                fig = plot_explanation(fig_rets, dash=True)
                return [childrend, fig]
            return [[], empty_bar()]


def html_dots(state, method="tsne", n_components=2, id_="interactions"):
    fig = dimension_reduction_plot(
        state, method_n=method, n_components=n_components, dash=True
    )
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


def draw_color(case, key_words):
    children = []
    pos_mark = "<POS>"
    pos_style = {"color": "red"}
    for word, field in key_words:
        case[field] = case[field].replace(word, f"{pos_mark}{word}")
    text = json.dumps(case, indent=2)
    tmp_text = []
    for part in text.split(" "):
        if pos_mark in part:
            children.append(" ".join(tmp_text))
            children.append(
                html.Span(f' {part.replace(pos_mark, "")} ', style=pos_style)
            )
            tmp_text.clear()
        else:
            tmp_text.append(part)
    children.append(" ".join(tmp_text))
    return html.P(children)
