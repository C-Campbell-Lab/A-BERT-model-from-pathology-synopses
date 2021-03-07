"""Data Transformation and Utility"""
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .domain import Cases, Label, LabelledCase, Labels, RawData
from .io_utils import dump_datazip, get_timestamp, load_json

random.seed(42)


def compose(case: dict, keep_key=False, shuffle=False) -> str:
    if keep_key:
        tmp = [f"{k}: {v}" for k, v in case.items()]
    else:
        tmp = list(case.values())
    if shuffle:
        random.shuffle(tmp)
    return " ".join(tmp)


def grouping_idx(y) -> Dict[str, list]:
    group_by = defaultdict(list)
    for idx, tags in enumerate(y):
        for tag in tags:
            group_by[tag].append(idx)
    return group_by


def count_tags(tags: Labels):
    return Counter(chain(*tags))


def count_token_len(cases: Cases):
    lens = list(map(lambda case: len(" ".join(case.values()).split(" ")), cases))
    return Counter(lens)


def unwrap_labelled_cases(labelled_cases: List[LabelledCase]):
    cases = [lc.text for lc in labelled_cases]
    tags = [lc.tag for lc in labelled_cases]
    return cases, tags


def label_to_tags(label: str):
    return refine_tag(label.split(";"))


def refine_tag(label: Label):
    return [
        tag
        for tag in add_acute_LL(
            list(
                map(
                    lambda x: tag_patch(edit_tag_str(x)),
                    label,
                )
            )
        )
        if tag != ""
    ]


def edit_tag_str(tag: str):
    typos = dict(
        [
            ("inadquate", "inadequate"),
            ("eosinophila", "eosinophilia"),
            ("hemophagoctyosis", "hemophagocytosis"),
            ("hypercellula", "hypercellular"),
            ("hypocellularr", "hypocellular"),
            ("lymphoma", "acute lymphoblastic leukemia"),
        ]
    )

    tag = tag.lower().strip().replace(".", "").replace("syndrome no", "syndrome")

    tag = typos.get(tag, tag)
    return tag


def tag_patch(tag: str):
    if tag == "plasma" or tag == "plasma cell disorder":
        return "plasma cell neoplasm"
    return tag


def add_acute_LL(tags: List[str]):
    """the tag \"lymphoproliferative disorder\"
    should be added to any case tagged as \"acute lymphoblastic leukemia\".

    Returns:
        [type]: [description]
    """
    w1 = "acute lymphoblastic leukemia"
    w2 = "lymphoproliferative disorder"
    if w1 in tags and w2 not in tags:
        return tags + [w2]
    return tags


def cases_minus(minuend: Cases, subtrahend: Cases):
    def not_same_content(case: dict) -> bool:
        return "".join(case.values()) not in used_cases

    used_cases = {"".join(case.values()) for case in subtrahend}
    assert "" not in used_cases, "has empty case in other cases"
    difference = list(filter(not_same_content, minuend))

    return difference


def labelled_cases_to_xy(labelled_cases: List[LabelledCase]):
    x = []
    y = []
    for labelled_case in labelled_cases:
        x.append(labelled_case.text)
        y.append(labelled_case.tag)
    return x, y


def xy_to_labelled_cases(x, y) -> List[LabelledCase]:
    return [LabelledCase(text, tag) for text, tag in zip(x, y)]


def split_and_dump_dataset(x, y, test_size=0.2):
    x_train_dict, x_test_dict, y_train_tags, y_test_tags = train_test_split(
        x, y, test_size=test_size
    )
    rd = RawData(x, y, x_train_dict, y_train_tags, x_test_dict, y_test_tags)
    zip_name = f"dataset{get_timestamp()}.zip"
    dump_datazip(rd, zip_name)
    return zip_name


def load_labelled_cases(path):
    records = load_json(path)
    labelled_cases = []
    for record in records:
        labelled_cases.append(
            LabelledCase(record["text"], label_to_tags(record["tag"]))
        )
    return labelled_cases


def train_test_split(x, y, test_size=0.2, train_first=True):
    if train_first:
        train_idx, test_idx = _train_first_split(y, test_size)
    else:
        train_idx, test_idx = _test_first_split(y, test_size)
    return (
        [x[idx] for idx in train_idx],
        [x[idx] for idx in test_idx],
        [y[idx] for idx in train_idx],
        [y[idx] for idx in test_idx],
    )


def _train_first_split(y, test_size):
    tag_stat = count_tags(y)
    keep_tags = get_rare_tags(tag_stat)
    train_idx = []
    test_idx = []
    left_idx = []
    # active learning period -> train_first
    for idx, tags in enumerate(y):
        if any(tag in keep_tags for tag in tags):
            train_idx.append(idx)
        else:
            left_idx.append(idx)
    random.shuffle(left_idx)
    bin_point = round(test_size * len(left_idx))
    train_idx.extend(left_idx[bin_point:])
    test_idx.extend(left_idx[:bin_point])
    return train_idx, test_idx


def _test_first_split(y, test_size):
    # evaluation learning period -> test_first
    tag_stat = count_tags(y)
    keep_tags = get_rare_tags(tag_stat)
    keep_tag_dict = {tag: 0 for tag in keep_tags}
    train_idx = []
    test_idx = []
    left_idx = []
    for idx, tags in enumerate(y):
        if any(tag in keep_tag_dict for tag in tags):
            test_idx.append(idx)
            for tag in tags:
                # make sure at least 3 cases are in test for each rare labels
                try:
                    keep_tag_dict[tag] += 1
                    if keep_tag_dict[tag] > 3:
                        del keep_tag_dict[tag]
                except KeyError:
                    pass
        else:
            left_idx.append(idx)
    random.shuffle(left_idx)
    bin_point = round(test_size * len(left_idx))
    train_idx.extend(left_idx[bin_point:])
    test_idx.extend(left_idx[:bin_point])
    return train_idx, test_idx


def get_rare_tags(tag_count: dict, thresh=1):
    return [tag for tag, count in tag_count.items() if count <= thresh]


def show_replica(cases: Cases):
    tmp = Counter("".join(case.values()) for case in cases)
    for k, v in tmp.items():
        if v > 1:
            print(k)


def get_Mlb(rawdata: RawData):
    return MultiLabelBinarizer().fit(rawdata.y_tags)


def topN(preds, classes, n=3):
    tops = np.argsort(preds)
    ret = []
    for i, top in enumerate(tops):
        sel_idx = top[-n:][::-1]
        ret.append(list(zip(classes[sel_idx], preds[i][sel_idx])))
    return ret


def rawdata_stat(rawdata: RawData):
    def counter_to_df(counter: dict):
        df = pd.DataFrame.from_dict(counter, orient="index")
        df.reset_index(inplace=True)
        df.columns = ["tag", "count"]
        return df

    train_counter = count_tags(rawdata.y_train_tags)
    test_counter = count_tags(rawdata.y_test_tags)
    print(f"Training Size: {len(rawdata.y_train_tags)}")
    print(f"Test Size: {len(rawdata.y_test_tags)}")
    print(f"Train Tag Number: {len(train_counter)}")
    print(f"Test Tag Number: {len(test_counter)}")

    train_tag_df = counter_to_df(train_counter)
    train_tag_df["for"] = "train"
    test_tag_df = counter_to_df(test_counter)
    test_tag_df["for"] = "test"

    tag_df = pd.concat([train_tag_df, test_tag_df])
    return tag_df


def refine_rawdata(rawdata: RawData):
    rawdata.y_tags = list(map(refine_tag, rawdata.y_tags))
    rawdata.y_test_tags = list(map(refine_tag, rawdata.y_test_tags))
    rawdata.y_train_tags = list(map(refine_tag, rawdata.y_train_tags))
    return rawdata


def adjust_normal(tags):
    w = "normal"
    for tag in tags:
        if len(tag) > 1 and w in tag:
            tag.remove(w)
