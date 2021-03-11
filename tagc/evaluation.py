from typing import List

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from tagc.io_utils import load_json


def expert_eval(sheet_csv, y_pred_: List[list] = None):
    p_df = pd.read_csv(sheet_csv).drop_duplicates(subset=["ID", "Judge"], keep="last")
    mlb = MultiLabelBinarizer()
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
    return out


def form_pred(eval_json):
    eval = load_json(eval_json)
    y_pred = []
    for item in eval:
        y_pred.append(
            [prob[0] for pred, prob in zip(item["pred"], item["prob"]) if pred]
        )
    return y_pred


def main(csv_p, eval_p="E:\\outputsS\\outputsS\\eval.json"):
    ret = expert_eval(csv_p, form_pred(eval_p))
    print(ret)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
