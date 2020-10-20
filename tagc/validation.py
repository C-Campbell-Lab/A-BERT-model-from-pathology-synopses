from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .data_utils import count_tags
from .domain import Mlb, RawData, States
from .model import StandaloneModel, label_output


def get_unlabelled_state(model: StandaloneModel, cases: list, mlb: Mlb, thresh=None):
    def tags_to_str(tags):
        return ", ".join(sorted(tags))

    k = len(cases)
    pooled_outputs = model.predict(cases, pooled_output=True)
    pred_tags = model.predict_tags(pooled_outputs, mlb, thresh=thresh)
    pred_tag_note = list(map(tags_to_str, pred_tags))
    index = list(range(k))
    tag_y = pred_tag_note
    tag_n = list(map(lambda tags: len(tags), pred_tags))
    from_ = ["unlabelled" for _ in range(k)]
    states = States(pooled_outputs, tag_y, index, tag_n, from_, pred_tag_note)
    return states


def get_tag_states(model: StandaloneModel, rawdata: RawData, mlb: Mlb, thresh=None):
    def tags_to_str(tags):
        return ", ".join(sorted(tags))

    x = rawdata.x_train_dict + rawdata.x_test_dict
    y = rawdata.y_train_tags + rawdata.y_test_tags
    index = list(range(len(rawdata.x_train_dict)))
    index.extend(range(len(rawdata.x_test_dict)))
    from_ = ["train" for _ in range(len(rawdata.x_train_dict))]
    from_.extend("test" for _ in range(len(rawdata.x_test_dict)))
    tag_n = list(map(lambda tags: len(tags), y))
    tag_y = list(map(tags_to_str, y))
    pooled_outputs = model.predict(x, pooled_output=True)
    pred_tags = model.predict_tags(x, mlb, thresh=thresh)
    pred_tag_note = list(map(tags_to_str, pred_tags))
    states = States(pooled_outputs, tag_y, index, tag_n, from_, pred_tag_note)
    return states


def dimension_reduction_plot(
    states: States, method_n="PCA", n_components=3, dash=False
):
    if method_n.lower() == "tsne":
        method = TSNE
    else:
        method = PCA
    dimension_reducer = method(n_components=n_components)
    result = dimension_reducer.fit_transform(states.data)
    if isinstance(dimension_reducer, PCA):
        print(
            f"Explained variation per principal component: {dimension_reducer.explained_variance_ratio_}"
        )
    df = pd.DataFrame(
        {
            "tag": states.tag,
            "index": states.index,
            "tag_num": states.tag_n,
            "from": states.from_,
            "pred_tag": states.pred_tag,
        }
    )
    for n in range(n_components):
        df[f"D{n+1}"] = result[:, n]

    if n_components == 3:
        fig = px.scatter_3d(
            df,
            x="D1",
            y="D2",
            z="D3",
            color="tag",
            symbol="tag_num",
            hover_data=["index", "from", "pred_tag"],
        )
    elif n_components == 2:
        fig = px.scatter(
            df,
            x="D1",
            y="D2",
            color="tag",
            symbol="tag_num",
            hover_data=["index", "from", "pred_tag"],
        )
    else:
        print("support only 2 or 3 dimension ploting")
        return
    fig.layout.update(showlegend=False)
    if dash:
        return fig, dimension_reducer
    fig.show()


def judge_on_tag(model: StandaloneModel, mlb: Mlb, rawdata: RawData, thresh=None):
    x = rawdata.x_test_dict
    y = rawdata.y_test_tags
    total_y = rawdata.y_tags
    pred_prob = model.predict(x)
    preds = label_output(pred_prob, thresh)
    y_vector = mlb.transform(y)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_vector, preds, average="micro"
    )
    mcm = metrics.multilabel_confusion_matrix(y_vector, preds)
    ability = list(map(compress, mcm))
    tag_count = count_tags(total_y)
    sample_sizes = [tag_count[class_] for class_ in mlb.classes_]
    performance = pd.DataFrame(
        {
            "Tag": mlb.classes_,
            "F1 Score": [pair[0] for pair in ability],
            "Testing Size": [pair[2] for pair in ability],
            "Sample Size": sample_sizes,
        }
    )
    performance["Training Size"] = (
        performance["Sample Size"] - performance["Testing Size"]
    )
    performance.sort_values(
        "F1 Score",
        inplace=True,
    )
    fig = px.scatter(
        performance, x="Tag", y="F1 Score", size="Training Size", color="Testing Size"
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                x=21,
                y=0.25,
                xref="x",
                yref="y",
                text=f"Precision: {precision:.03f}",
                showarrow=False,
            ),
            dict(
                x=21,
                y=0.2,
                xref="x",
                yref="y",
                text=f"Recall: {recall:.03f}",
                showarrow=False,
            ),
            dict(
                x=21,
                y=0.15,
                xref="x",
                yref="y",
                text=f"F1 Score: {f1:.03f}",
                showarrow=False,
            ),
        ],
    )
    return fig


def compress(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    amount = fn + tp
    if amount == 0:
        f1 = 0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return (f1, tp, amount)


def summary(cases, true_tags, pred_tags):
    example = []
    judges = []
    less_tag_num = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    more_tag_num = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    equal_tag_num = defaultdict(lambda: defaultdict(int))
    data = {
        "less": less_tag_num,
        "more": more_tag_num,
        "equal": equal_tag_num,
    }
    for case, pred_tag, true_tag in zip(cases, pred_tags, true_tags):
        num_tag = len(true_tag)
        pred_num_tag = len(pred_tag)
        corr = sum(tag in true_tag for tag in pred_tag)
        if pred_num_tag < num_tag:
            judge = f"Less: {corr} in {pred_num_tag} tags correct"
            data["less"][num_tag][pred_num_tag - num_tag][corr] += 1
        elif pred_num_tag > num_tag:
            judge = f"More: {corr} in {pred_num_tag} tags correct"
            data["more"][num_tag][pred_num_tag - num_tag][corr] += 1
        else:
            if corr == pred_num_tag:
                judge = "Correct"
            else:
                judge = f"Equal: {corr} in {pred_num_tag} tags correct"
            data["equal"][num_tag][corr] += 1

        example.append((case, "; ".join(pred_tag), "; ".join(true_tag), judge))
        judges.append(judge)

    return example, Counter(judges), data


def judge_to_df(on_tag_num):
    labels = []
    tag_num = []
    num = []
    for k, v in on_tag_num.items():
        counter = Counter(v)
        for corr_class, corr_num in counter.items():
            tag_num.append(k)
            labels.append(f"{corr_class} Correct")
            num.append(corr_num)
    return pd.DataFrame({"Tag Num": tag_num, "Count": num, "Category": labels})
