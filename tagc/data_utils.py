import json
import random
import time
from collections import Counter, defaultdict
from itertools import chain
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from .domain import DATAFILE, Cases, LabelledCase, RawData

random.seed(42)


def grouping_idx(y) -> Dict[str, list]:
    group_by = defaultdict(list)
    for idx, tags in enumerate(y):
        for tag in tags:
            group_by[tag].append(idx)
    return group_by


def count_tags(tags: List[List[str]]):
    return Counter(chain(*tags))


def count_token_len(texts: List[str]):
    lens = list(map(lambda text: len(text.split(" ")), texts))
    return Counter(lens)


def load_json(path):
    with open(path, "r") as js_:
        return json.load(js_)


def dump_json(path, obj):
    with open(path, "w") as js_:
        json.dump(obj, js_)


def load_labelled_cases(path):
    records = load_json(path)
    labelled_cases = []
    for record in records:
        labelled_cases.append(
            LabelledCase(record["text"], label_to_tags(record["tag"]))
        )
    return labelled_cases


def unwrap_labelled_cases(labelled_cases: List[LabelledCase]):
    cases = [lc.text for lc in labelled_cases]
    tags = [lc.tag for lc in labelled_cases]
    return cases, tags


def label_to_tags(label: str):
    tmp_tags = list(
        map(
            lambda x: tag_patch(
                x.lower()
                .strip()
                .replace(".", "")
                .replace("syndrome no", "syndrome")
                .replace("inadquate", "inadequate")
            ),
            label.split(";"),
        )
    )
    refine_tags = add_acute_LL(tmp_tags)
    return refine_tags


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


def dump_labelled_cases(labelled_cases: List[LabelledCase], path: str):
    obj = list(map(LabelledCase.serialize, labelled_cases))
    dump_json(path, obj)


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


def split_and_dump_dataset(x, y):
    x_train_dict, x_test_dict, y_train_tags, y_test_tags = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    rd = RawData(x, y, x_train_dict, y_train_tags, x_test_dict, y_test_tags)
    zip_name = f"dataset{get_timestamp()}.zip"
    dump_datazip(rd, zip_name)
    return zip_name


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def show_replica(cases: Cases):
    tmp = Counter("".join(case.values()) for case in cases)
    for k, v in tmp.items():
        if v > 1:
            print(k)


def load_datazip(datazip_path: str, datafile: dict = DATAFILE):
    with ZipFile(datazip_path, "r") as datazip:
        tmp = []
        for f_name in datafile.values():
            with datazip.open(f_name, "r") as file:
                tmp.append(json.loads(file.read().decode("utf-8")))
        return RawData(*tmp)


def dump_datazip(rawdata: RawData, zip_name="dataset1.1.zip"):
    with ZipFile(zip_name, "w") as datazip:
        for name, data in rawdata:

            with datazip.open(f"{name}.json", "w") as file:
                file.write(json.dumps(data).encode("utf-8"))


def get_Mlb(rawdata: RawData):
    return MultiLabelBinarizer().fit(rawdata.y_tags)


def topN(preds, classes, n=3):
    tops = np.argsort(preds)
    ret = []
    for i, top in enumerate(tops):
        sel_idx = top[-n:][::-1]
        ret.append(list(zip(classes[sel_idx], preds[i][sel_idx])))
    return ret
