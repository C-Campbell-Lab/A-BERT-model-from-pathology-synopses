import json
import random
from collections import Counter
from itertools import chain
from typing import List

from sklearn.model_selection import train_test_split

from .domain import LabelledCase

random.seed(42)


class DataProfile:
    def count_tags(self, tags: List[List[str]]):
        return Counter(chain(*tags)).most_common()

    def count_token_len(self, texts: List[str]):
        lens = list(map(lambda text: len(text.split(" ")), texts))
        return Counter(lens).most_common()


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


def labelled_cases_to_xy(labelled_cases: List[LabelledCase]):
    x = []
    y = []
    for labelled_case in labelled_cases:
        x.append(labelled_case.text)
        y.append(labelled_case.tag)
    return x, y


def xy_to_labelled_cases(x, y):
    return [LabelledCase(text, tag) for text, tag in zip(x, y)]


def split_and_dump_xy(x, y):
    x_train_dict, x_test_dict, y_train_tags, y_test_tags = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    dump_json("x.json", x)
    dump_json("y.json", y)
    dump_json("x_test_dict.json", x_test_dict)
    dump_json("y_test_tags.json", y_test_tags)
    dump_json("x_train_dict.json", x_train_dict)
    dump_json("y_train_tags.json", y_train_tags)


def show_replica(cases: List[dict]):
    tmp = Counter("".join(case.values()) for case in cases)
    for k, v in tmp.items():
        if v > 1:
            print(k)


def get_unlabelled(all_cases: List[dict], other_cases: List[dict]):
    def not_same_content(case: dict) -> bool:
        return "".join(case.values()) not in used_cases

    used_cases = {"".join(case.values()) for case in other_cases}
    assert "" not in used_cases, "has empty case in other cases"
    unlabelled_cases = list(filter(not_same_content, all_cases))

    return unlabelled_cases


def edit_review(old_cases: List[LabelledCase], reviews: List[LabelledCase]):
    cases_copy = old_cases.copy()
    review_map = {"".join(review.text.values()): review.tag for review in reviews}
    for idx, case in enumerate(old_cases):
        key = "".join(case.text.values())
        if key in review_map:
            cases_copy[idx].tag = review_map[key]
    return cases_copy
