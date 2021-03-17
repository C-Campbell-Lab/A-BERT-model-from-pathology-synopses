"""Used for co-occurrence count, Chord Diagram is made by chordDia.R"""

import numpy as np
import pandas as pd
from tagc.io_utils import load_json
from itertools import combinations

acronyms = {
    "acute leukemia": "acute leukemia",
    "acute lymphoblastic leukemia": "ALL",
    "acute myeloid leukemia": "AML",
    "acute promyelocytic leukemia": "APL",
    "basophilia": "basophilia",
    "chronic myeloid leukemia": "CML",
    "eosinophilia": "eosinophilia",
    "erythroid hyperplasia": "EHypr",
    "granulocytic hyperplasia": "GHypr",
    "hemophagocytosis": "HPCYT",
    "hypercellular": "hyper",
    "hypocellular": "hypo",
    "inadequate": "inadequate",
    "iron deficiency": "ID",
    "lymphoproliferative disorder": "LPD",
    "mastocytosis": "mastocytosis",
    "metastatic": "metastatic",
    "monocytosis": "monocytosis",
    "myelodysplastic syndrome": "MDS",
    "myeloproliferative neoplasm": "MPN",
    "normal": "normal",
    "plasma cell neoplasm": "PCN",
}


def eval_to_mat(eval_p="eval.json", dst="mat_abbr.csv"):
    eval_rets = load_json(eval_p)
    tags = [item[0] for item in eval_rets[0]["prob"]]
    tag_df = to_tag_df(eval_rets, tags)
    adj_df = form_adj_df(tag_df)
    mat_df = form_mat_df(adj_df)
    mat_df.rename(acronyms, axis=1, inplace=True)
    mat_df.rename(acronyms, axis=0, inplace=True)
    mat_df.to_csv(dst)


def to_tag_df(eval_rets, tags):
    tag_num_col = []
    tag_col = []
    for ret in eval_rets:
        tag = [tags[idx] for idx, sel in enumerate(ret["pred"]) if sel]
        tag_num_col.append(len(tag))
        tag_col.append(", ".join(sorted(tag)))
    df = pd.DataFrame({"tag_num": tag_num_col, "tag": tag_col}).reset_index()
    return df


def form_adj_df(tag_df):
    count_df = tag_df.groupby(["tag_num", "tag"])["index"].count()
    start = []
    end = []
    weights = []
    tag_nums = tag_df.tag_num.unique()
    for tag_num in filter(lambda x: x != 1, tag_nums):
        tmp = count_df.loc[tag_num]
        for label, weight in tmp.iteritems():
            for a, b in combinations(label.split(", "), 2):
                start.append(a)
                end.append(b)
                weights.append(weight)
    return pd.DataFrame({"start": start, "end": end, "weight": weights})


def form_mat_df(adj_df):
    tags_list = sorted(np.unique(adj_df.loc[:, ["start", "end"]].values))
    size = len(tags_list)
    matrix = np.zeros((size, size), dtype=int)
    loc_dict = {t: i for i, t in enumerate(tags_list)}
    for _, items in adj_df.iterrows():
        i = loc_dict[items[0]]
        j = loc_dict[items[1]]
        matrix[i][j] += items[2]
        matrix[j][i] += items[2]
    df = pd.DataFrame(matrix, columns=tags_list, index=tags_list)
    return df


if __name__ == "__main__":
    from fire import Fire

    Fire(eval_to_mat)
