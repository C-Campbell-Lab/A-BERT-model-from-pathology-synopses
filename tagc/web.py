import os

import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MultiLabelBinarizer

from . import web_utils
from .io_utils import load_datazip, load_json, prepare_dataset, prepare_model
from .mask_explain import MaskExplainer, plot_explanation
from .model import StandaloneModel
from .validation import dimension_reduction_plot, get_tag_states


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
        dot_id = "t-sne"
        case_id = "cases"
        mask_id = "mask"
        checkbox_id = "checkbox"
        submit_idx = "submit"
        app.layout = html.Div(
            [
                web_utils.html_input(id_=input_id),
                html.H2("t-SNE"),
                web_utils.html_dots(dot_id, self.fig),
                html.H2("Explanation"),
                web_utils.html_case(id_=case_id),
                web_utils.html_mask(id_=mask_id),
                html.H2("Check the incorrect predictions"),
                web_utils.html_checkbox(id_=checkbox_id),
                web_utils.html_submit(id_=submit_idx),
            ]
        )

        @app.callback(
            [
                Output(case_id, "children"),
                Output(mask_id, "figure"),
                Output(checkbox_id, "options"),
                Output(submit_idx, "disabled"),
            ],
            [
                Input(input_id, "value"),
                Input(dot_id, "clickData"),
            ],
        )
        def update_output_div(input_value, clickData):
            if input_value is not None:
                return self._case_plot({"COMMENT": input_value})
            elif clickData is not None:
                return self._display_click_data(clickData)
            return [[], web_utils.empty_bar(), [], True]

        @app.callback(
            [
                Output(dot_id, "clickData"),
                Output(input_id, "value"),
                Output(checkbox_id, "value"),
            ],
            [Input(submit_idx, "n_clicks")],
            [State(checkbox_id, "value")],
        )
        def update_output(n_clicks, value):
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate("cancel the callback")
            else:
                print(
                    'The input value was "{}" and the button has been clicked {} times'.format(
                        value, n_clicks
                    )
                )
                return None, None, []

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
        options = []
        if len(rets) == 0:
            childrend.append(html.H3("No confidence in any predictions"))
            childrend.append(web_utils.dict_to_str(case))
            disabled_submit = True
            fig = web_utils.empty_bar()
        else:
            for ret in rets:
                options.append({"label": ret.tag, "value": ret.tag})
                importance = ret.importance
                pos_key_marks = [p[0] for p in importance][:5]
                childrend.append(html.H3(ret.tag))
                childrend.append(web_utils.draw_color(case, pos_key_marks))

            disabled_submit = False
            fig = plot_explanation(rets, dash=True)

        return [childrend, fig, options, disabled_submit]
