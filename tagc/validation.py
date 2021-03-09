from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .data_utils import count_tags, rawdata_stat
from .domain import Mlb, RawData, States
from .io_utils import dump_json
from .model import StandaloneModel, label_output
from .visualization import (
    plot_num_performance,
    plot_summary,
    plot_tag_performance,
    plot_tag_stat,
)


def eval_model(model, ds, repeat, mlb, output_p, marker):
    tag_stat = rawdata_stat(ds)
    tag_stat.to_csv(f"{output_p}/{marker}_{repeat}_data_stat.csv")
    fig = plot_tag_stat(tag_stat)
    fig.update_layout(
        width=1280,
        height=600,
    )
    fig.write_image(f"{output_p}/{marker}_{repeat}_data_stat.pdf")
    performance, metric, pred_tags = judge_on_tag(model, mlb, ds, n=repeat)
    dump_json(f"{output_p}/{marker}_{repeat}_overall.json", metric)
    performance.to_csv(f"{output_p}/{marker}_{repeat}_Perf_tag.csv")
    fig = plot_tag_performance(performance, metric, auc=False)
    fig.write_image(f"{output_p}/{marker}_{repeat}_Perf_tag.pdf")

    examples, judge_count, data, df = summary(
        ds.x_test_dict,
        ds.y_test_tags,
        pred_tags,
    )
    df.to_csv(f"{output_p}/{marker}_{repeat}_summary.csv")
    return examples, judge_count, data, df


def extra_summary_suite(examples, data, df, output_p):
    df.to_csv(f"{output_p}/summary.csv")
    fig = plot_summary(data)
    fig.write_image(f"{output_p}/fig3b_Pie.pdf")
    performance_summary = judge_on_summary(df)
    fig = plot_num_performance(performance_summary)
    fig.write_image(f"{output_p}/fig3c_Perf_sum.pdf")
    review = []
    for case, pred_tag, true_tag, judge in examples:
        if "Label" in judge:
            review.append({"text": case, "pred_tag": pred_tag, "tag": true_tag})
    dump_json(f"{output_p}/review.json", review)


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


def judge_on_tag(
    model: StandaloneModel, mlb: Mlb, rawdata: RawData, thresh=None, n=-1, micro=True
):
    x = rawdata.x_test_dict
    y = rawdata.y_test_tags
    total_y = y + rawdata.y_train_tags
    if n == -1:
        pred_prob = model.predict(x)
    else:
        pred_prob = model.over_predict(x, n=n)

    preds = label_output(pred_prob, thresh)
    y_vector = mlb.transform(y)
    pred_tags = mlb.inverse_transform(preds)

    # Fileter
    class_num = len(mlb.classes_)
    sel_tag_idx = [i for i in range(class_num) if len(np.unique(y_vector[:, i])) != 1]
    y_vector = y_vector[:, sel_tag_idx]
    pred_prob = pred_prob[:, sel_tag_idx]
    preds = preds[:, sel_tag_idx]
    sel_class = [mlb.classes_[i] for i in sel_tag_idx]

    # Computation
    if micro:
        average_auc = metrics.roc_auc_score(
            y_vector, pred_prob, average="micro", multi_class="ovr"
        )
    else:
        average_auc = metrics.roc_auc_score(
            y_vector, pred_prob, average="macro", multi_class="ovr"
        )

    aucs = [
        metrics.roc_auc_score(y_vector[:, i], pred_prob[:, i])
        for i in range(len(sel_tag_idx))
    ]

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_vector, preds, average="micro"
    )
    mcm = metrics.multilabel_confusion_matrix(y_vector, preds)
    ability = list(map(compress, mcm))
    tag_count = count_tags(total_y)
    sample_sizes = [tag_count[class_] for class_ in sel_class]
    # Aggregation
    performance = pd.DataFrame(
        {
            "Tag": sel_class,
            "Precision": [pair[0] for pair in ability],
            "Recall": [pair[1] for pair in ability],
            "F1 Score": [pair[2] for pair in ability],
            "Testing Size": [pair[3] for pair in ability],
            "Sample Size": sample_sizes,
            "TN": [pair[4] for pair in ability],
            "FP": [pair[5] for pair in ability],
            "FN": [pair[6] for pair in ability],
            "TP": [pair[7] for pair in ability],
            "AUC": aucs,
        }
    )
    performance["Training Size"] = (
        performance["Sample Size"] - performance["Testing Size"]
    )

    metric_ret = {
        "precision": precision,
        "recall": recall,
        "F1 Score": f1,
        "AUC": average_auc,
    }
    return performance, metric_ret, pred_tags


def compress(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    amount = fn + tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, amount)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return (precision, recall, f1, amount, tn, fp, fn, tp)


def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


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


def judge_on_num(
    model: StandaloneModel, mlb: Mlb, rawdata: RawData, thresh=None, n=None
):
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
        if n is not None:
            pred_prob = model.over_predict(num_x, n=n)
        else:
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
