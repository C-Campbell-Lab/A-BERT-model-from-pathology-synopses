from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import miniball

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


def dimension_reduction(states: States, method_n="PCA", n_components=3):
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
    return df


def dimension_reduction_plot(df, n_components=3):
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
    return fig


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
            "Precision": [pair[0] for pair in ability],
            "Recall": [pair[1] for pair in ability],
            "F1 Score": [pair[2] for pair in ability],
            "Testing Size": [pair[3] for pair in ability],
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
    return performance, {"precision": precision, "recall": recall, "f1": f1}


def compress(cm):
    def safe_divide(a, b):
        if b == 0:
            return 0
        return a / b

    tn, fp = cm[0]
    fn, tp = cm[1]
    amount = fn + tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, amount)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return (precision, recall, f1, amount)


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
    corrects = []
    pred_tag_numbers = []
    tag_numbers = []
    for case, pred_tag, true_tag in zip(cases, pred_tags, true_tags):
        tag_num = len(true_tag)
        pred_tag_num = len(pred_tag)
        corr = sum(tag in true_tag for tag in pred_tag)
        if pred_tag_num < tag_num:
            judge = f"Less: {corr}/{pred_tag_num} correct. Label:{tag_num} tags"
            data["less"][tag_num][pred_tag_num - tag_num][corr] += 1
        elif pred_tag_num > tag_num:
            judge = f"More: {corr}/{pred_tag_num} correct. Label:{tag_num} tags"
            data["more"][tag_num][pred_tag_num - tag_num][corr] += 1
        else:
            if corr == pred_tag_num:
                judge = "correct"
            else:
                judge = f"Equal: {corr}/{pred_tag_num} correct. Label:{tag_num} tags"
            data["equal"][tag_num][corr] += 1

        corrects.append(corr)
        pred_tag_numbers.append(pred_tag_num)
        tag_numbers.append(tag_num)

        example.append((case, "; ".join(pred_tag), "; ".join(true_tag), judge))
        judges.append(judge)

    df = pd.DataFrame(
        {
            "Correct Count": corrects,
            "Pred Tag Number": pred_tag_numbers,
            "Tag Number": tag_numbers,
        }
    )
    return (example, Counter(judges), data, df)


def judge_on_num(model: StandaloneModel, mlb: Mlb, rawdata: RawData, thresh=None):
    x = rawdata.x_test_dict
    y = rawdata.y_test_tags
    tag_num_map = defaultdict(list)
    for idx, tag in enumerate(y):
        tag_num_map[len(tag)].append(idx)

    tag_nums = []
    sizes = []
    precisions = []
    recalls = []
    f1s = []

    for tag_num, indexes in tag_num_map.items():
        num_x = [x[idx] for idx in indexes]
        num_y = [y[idx] for idx in indexes]
        pred_prob = model.predict(num_x)
        preds = label_output(pred_prob, thresh)
        y_vector = mlb.transform(num_y)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_vector, preds, average="micro"
        )
        tag_nums.append(tag_num)
        sizes.append(len(indexes))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    df = pd.DataFrame(
        {
            "Tag Number": tag_nums,
            "Count": sizes,
            "F1 Score": f1s,
            "Recall": recalls,
            "Precision": precisions,
        }
    )
    df.index = df["Tag Number"]
    return df


def judge_on_summary(summary_df: pd.DataFrame):
    def gb(df, c):
        df_ = df.copy()
        tmp_c = c + "_"
        df_[tmp_c] = df[c]
        return df_.groupby(tmp_c).sum()

    tc = gb(summary_df, "Tag Number")
    tc["Recall"] = tc["Correct Count"] / tc["Tag Number"]
    tc["Precision"] = tc["Correct Count"] / tc["Pred Tag Number"]
    tc["F1 Score"] = (
        2 * tc["Recall"] * tc["Precision"] / (tc["Recall"] + tc["Precision"])
    )
    tc["Count"] = tc["Tag Number"] / tc.index
    return tc


def merge_dimension(df: pd.DataFrame):
    collector = defaultdict(list)
    for row in df.iterrows():
        items = row[1]
        label: str = items[0]
        for tag in label.split(", "):
            collector[tag].append(items[-2:])

    d1s = []
    d2s = []
    counts = []
    tags = []
    r_square = []
    for tag, locs in collector.items():
        count = len(locs)
        counts.append(count)
        locs = np.array(locs, dtype="float")
        C, r2 = miniball.get_bounding_ball(locs)
        r_square.append(r2)
        d1s.append(C[0])
        d2s.append(C[1])
        tags.append(tag)

    return pd.DataFrame(
        {"Count": counts, "Tag": tags, "D1": d1s, "D2": d2s, "R square": r_square}
    )
