from itertools import chain
from typing import List

import pandas as pd
import plotly.express as px
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .data_utils import count_tags
from .domain import Case, States
from .model import StandaloneModel

# dimension_reduction_plot


def get_one_tag_states(model: StandaloneModel, x: List[Case], y: List[List[str]]):
    pooled_outputs = model.predict(x, pooled_output=True)
    tmp_y = list(filter(lambda pair: len(pair[1]) == 1, enumerate(y)))
    index = [x[0] for x in tmp_y]
    one_tag_y = list(chain(*(x[1] for x in tmp_y)))
    one_tag_x = pooled_outputs[index]
    states = States(one_tag_x, one_tag_y, index)
    return states


def dimension_reduction_plot(states: States, method_n="PCA", n_components=3):
    if method_n.lower() == "tsne":
        method = PCA
    else:
        method = TSNE
    dimension_reducer = method(n_components=n_components)
    result = dimension_reducer.fit_transform(states.data)
    if isinstance(dimension_reducer, PCA):
        print(
            f"Explained variation per principal component: {dimension_reducer.explained_variance_ratio_}"
        )
    df = pd.DataFrame({"tag": states.tag, "index": states.index})
    for n in range(n_components):
        df[f"D{n+1}"] = result[:, n]

    if n_components == 3:
        fig = px.scatter_3d(
            df, x="D1", y="D2", z="D3", color="tag", hover_data=["index"]
        )
        fig.show()
    elif n_components == 2:
        fig = px.scatter(df, x="D1", y="D2", color="tag", hover_data=["index"])
        fig.show()
    else:
        print("support only 2 or 3 dimension ploting")


# metrics for each tag


def judge_on_tag(model: StandaloneModel, x, y, mlb, total_y):
    pred_prob = model.predict(x)
    preds = pred_prob >= 0.5
    mcm = metrics.multilabel_confusion_matrix(mlb.transform(y), preds > 0.5)
    ability = list(map(compress, mcm))
    tag_count = count_tags(total_y)
    sample_sizes = [tag_count[class_] for class_ in mlb.classes_]
    performance = pd.DataFrame(
        {
            "Tag": mlb.classes_,
            "Acc": [pair[0] for pair in ability],
            "Num": [pair[2] for pair in ability],
            "Sample_Size": sample_sizes,
        }
    )
    fig = px.scatter(performance, x="Tag", y="Acc", size="Sample_Size", color="Num")
    fig.show()


def compress(cm):
    err, corr = cm[1]
    amount = err + corr
    if amount == 0:
        return (0, corr, amount)
    return (corr / amount, corr, amount)
