import json
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import requests
from dash.dependencies import Input, Output
from sklearn.preprocessing import MultiLabelBinarizer

from .io_utils import load_datazip
from .model import StandaloneModel
from .validation import dimension_reduction_plot, get_tag_states


def load_state(state_p: str):
    with open(state_p, "rb") as plk:
        state = pickle.load(plk)
    return state


def download_file(url, out_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return out_path


class Server:
    def __init__(self, state=None):
        self.rawdata = load_datazip("data/dataset.zip")
        if state is None:
            self.init_state()
        else:
            self.state = state

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def init_state(self):
        self.model = StandaloneModel.from_path("CP")
        self.mlb = MultiLabelBinarizer().fit(self.rawdata.y_tags)
        self.state = get_tag_states(self.model, self.rawdata, self.mlb)

    def plot(self, method="tsne", n_components=2):
        styles = {
            "pre": {
                "overflowX": "scroll",
            }
        }
        fig = dimension_reduction_plot(
            self.state, method_n=method, n_components=n_components, dash=True
        )
        app = self.app

        app.layout = html.Div(
            [
                dcc.Graph(id="basic-interactions", figure=fig),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            [
                                dcc.Markdown(
                                    """
                **Case Content**

                Click on points in the graph.
            """
                                ),
                                html.Pre(id="click-data", style=styles["pre"]),
                            ],
                            className="eleven columns",
                        ),
                    ],
                ),
            ]
        )

        @app.callback(
            Output("click-data", "children"), [Input("basic-interactions", "clickData")]
        )
        def display_click_data(clickData):
            if clickData is not None:
                customdata = clickData["points"][0]["customdata"]
                idx = customdata[0]
                from_ = customdata[1]
                data = self.rawdata.retrive(from_, idx)
                return json.dumps(data, indent=2)
            # return json.dumps(clickData, indent=2)
