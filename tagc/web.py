import json
import os
import shutil
from dataclasses import dataclass

import dash
import dash_html_components as html
import requests
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MultiLabelBinarizer

from . import io_utils, web_utils
from .mask_explain import MaskExplainer, plot_explanation
from .model import StandaloneModel
from .validation import dimension_reduction_plot
from tagc.validation import dimension_reduction

URL = "https://gosheet-bqjlnzid4q-uc.a.run.app/add"


@dataclass
class WebConfig:
    dataset_p: str
    model_p: str
    unlabelled_p: str
    state_p: str

    def init_static(self):
        if not os.path.exists(self.dataset_p):
            io_utils.prepare_dataset(self.dataset_p)
        if not os.path.exists(self.model_p):
            io_utils.prepare_model(self.model_p)
        if not os.path.exists(self.unlabelled_p):
            io_utils.prepare_unlabelled(self.unlabelled_p)
        if not os.path.exists(self.state_p):
            io_utils.prepare_state(self.state_p)

    def reset(self):
        shutil.rmtree(self.model_p)
        os.remove(self.dataset_p)
        os.remove(self.unlabelled_p)
        os.remove(self.state_p)


class Server:
    def __init__(self, web_config: WebConfig):
        self.dataset_p = web_config.dataset_p
        self.model_p = web_config.model_p
        self.unlabelled_p = web_config.unlabelled_p
        self.state_p = web_config.state_p

        self.init_state()
        self.init_plot()

        self.case_str = ""
        self.tag_str = ""

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def init_state(self):
        self.rawdata = io_utils.load_datazip(self.dataset_p)
        self.unlabelled = io_utils.load_json(self.unlabelled_p)

        self.model = StandaloneModel.from_path(self.model_p)
        self.mlb = MultiLabelBinarizer().fit(self.rawdata.y_tags)
        self.mask_explainer = MaskExplainer(self.mlb)
        self.state = io_utils.load_state(self.state_p)

    def init_plot(self):
        state_df = dimension_reduction(self.state, method_n="tsne", n_components=2)
        self.fig = dimension_reduction_plot(state_df, n_components=2)

    def plot(self):
        app = self.app
        input_id = "input"
        input_submit_id = "input_submit"
        dot_id = "t-sne"
        case_id = "cases"
        mask_id = "mask"
        checkbox_id = "checkbox"
        submit_idx = "submit"
        app.layout = html.Div(
            [
                web_utils.html_input(id_=input_id),
                web_utils.html_submit(id_=input_submit_id),
                html.H2("t-SNE"),
                web_utils.html_dots(dot_id, self.fig),
                html.H2("Explanation"),
                web_utils.html_case(id_=case_id),
                web_utils.html_mask(id_=mask_id),
                html.H2("Check the correct predictions"),
                web_utils.html_checkbox(id_=checkbox_id),
                web_utils.html_submit(id_=submit_idx),
            ]
        )

        @app.callback(
            [
                Output(case_id, "children"),
                Output(mask_id, "figure"),
                Output(mask_id, "style"),
                Output(checkbox_id, "options"),
                Output(checkbox_id, "value"),
                Output(submit_idx, "disabled"),
            ],
            [
                Input(input_submit_id, "n_clicks"),
                Input(dot_id, "clickData"),
            ],
            [State(input_id, "value")],
        )
        def update_output_div(n_clicks, clickData, input_value):
            if clickData is not None:
                return self._display_click_data(clickData)

            elif input_value is not None and n_clicks is not None:
                return self._case_plot({"COMMENT": input_value})

            return [[], web_utils.empty_bar(), {"display": "none"}, [], [], True]

        @app.callback(
            [
                Output(dot_id, "clickData"),
                Output(input_id, "value"),
                Output(input_submit_id, "n_clicks"),
            ],
            [Input(submit_idx, "n_clicks")],
            [State(checkbox_id, "value")],
        )
        def update_output(n_clicks, value):
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate("cancel the callback")
            else:
                judge = {
                    "text": self.case_str,
                    "rets": self.tag_str,
                    "mistakes": json.dumps(value),
                }
                requests.post(URL, json=judge)

                return None, None, None

    def _display_click_data(self, clickData):

        customdata = clickData["points"][0]["customdata"]
        idx = customdata[0]
        from_ = customdata[1]
        pred_tag = customdata[2]
        if from_ == "unlabelled":
            case = self.unlabelled[idx]
        else:
            data = self.rawdata.retrive(from_, idx)
            case = data["text"]
        return self._case_plot(case, pred_tag)

    def _case_plot(self, case, pred_tag):
        childrend = []
        self.case_str = json.dumps(case)
        self.tag_str = json.dumps(pred_tag)
        prob = self.model.predict_prob([case], self.mlb)[0]

        options = [{"label": f"{c}: {p:.02f}", "value": c} for (c, p) in prob]
        if len(pred_tag) == 0:
            childrend.append(html.H3("No confidence in any predictions"))
            childrend.append(web_utils.dict_to_str(case))
            max_prob = max(prob, key=lambda x: x[1])
            values = [max_prob[0]]
            disabled_submit = False
            fig = web_utils.empty_bar()
            style = {"display": "none"}
        else:
            rets = self.mask_explainer.explain(self.model, case)
            values = [ret.tag for ret in rets]
            for ret in rets:
                importance = ret.importance
                pos_key_marks = [p[0] for p in importance][:5]
                childrend.append(html.H3(ret.tag))
                childrend.append(web_utils.draw_color(case, pos_key_marks))
            disabled_submit = False
            fig = plot_explanation(rets, dash=True)
            style = {"display": "block"}
        return [childrend, fig, style, options, values, disabled_submit]
