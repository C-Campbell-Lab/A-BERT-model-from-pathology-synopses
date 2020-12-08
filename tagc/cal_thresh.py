import numpy as np
from numpy import argmax
from sklearn.metrics import precision_recall_curve

from tagc.io_utils import dump_json


def cal_f1(precision, recall):
    a = 2 * precision * recall
    b = precision + recall
    fscore = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return fscore


def analysis_kf(rets, mlb, out_folder=""):
    labels = []

    for k_fold in rets:
        k_probs = k_fold["prob"]
        labels.extend(k_fold["tag"])

    labels = mlb.transform(labels)
    probs = np.zeros_like(labels, dtype=np.float)

    thresh = {}

    for tag_idx, tag in enumerate(mlb.classes_):
        tag_probs = []
        for k_fold in rets:
            k_probs = k_fold["prob"]
            tag_probs.extend(float(prob[tag_idx][1]) for prob in k_probs)
        tag_labels = labels[:, tag_idx]
        tag_probs = np.array(tag_probs)

        probs[:, tag_idx] = tag_probs

        precision, recall, thresholds = precision_recall_curve(tag_labels, tag_probs)
        fscore = cal_f1(precision, recall)
        # locate the index of the largest f score
        ix = argmax(fscore)
        print(
            "{}: Best Threshold={:f}, F-Score={:.3f}".format(
                tag, thresholds[ix], fscore[ix]
            )
        )
        thresh[tag_idx] = thresholds[ix]

    thresh_json = []
    for k, v in thresh.items():
        thresh_json.append((mlb.classes_[k], str(v)))
    dump_json(out_folder + "thresh_info.json", thresh_json)

    thresh_v = [v if v < 0.5 else 0.5 for v in thresh.values()]
    dump_json(out_folder + "thresh.json", thresh_v)
    # return thresh_json, thresh_v
