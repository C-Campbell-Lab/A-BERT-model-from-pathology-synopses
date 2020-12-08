from os.path import join
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List
from tagc.io_utils import load_json


def active_eval(sheet_csv, y_pred_: List[list] = None):
    p_df = pd.read_csv(sheet_csv).drop_duplicates(subset=["ID", "Judge"], keep="last")
    mlb = MultiLabelBinarizer()
    # mlb.fit(p_df["eval"].map(lambda x: x.split(", ")).tolist())
    mlb.fit(y_pred_)
    p_df = p_df.groupby("Judge").agg(list)
    out = {}
    all_evals = []

    for judge, df in p_df.iterrows():

        y_true = list(map(lambda x: x.split(", "), df["eval"]))
        ids = df["ID"]
        if y_pred_ is None:
            y_pred = list(map(lambda x: x.split(", "), df["pred"]))
        else:
            y_pred = [y_pred_[id_] for id_ in ids]

        y_true_trans = mlb.transform(y_true)
        y_pred_trans = mlb.transform(y_pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_true_trans, y_pred_trans, average="micro"
        )

        all_evals.append(y_true_trans)
        out[judge] = {"precision": precision, "recall": recall, "f1": f1}
    # cut = min(len(e) for e in all_evals)
    # p1 = all_evals[0][:cut]
    # p2 = all_evals[1][:cut]
    # print(p1.shape, p2.shape)
    # reliability_data = [p1, p2]
    # print(krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal'))

    return out


def form_pred(eval_json):
    eval = load_json(eval_json)
    y_pred = []
    for item in eval:
        y_pred.append(
            [prob[0] for pred, prob in zip(item["pred"], item["prob"]) if pred]
        )
    return y_pred


if __name__ == "__main__":
    base = "E:/Coding/scholar/active"
    sheet_csv = join(".", "prediction judgement - Sheet1.csv")
    print(
        active_eval(sheet_csv, form_pred("E:\\outputsS\\outputsS\\eval.json")),
    )
