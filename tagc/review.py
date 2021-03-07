"""Active learning: CRL sampling implementation"""

import random
from typing import Dict, List

from . import data_utils, io_utils
from .domain import Case, LabelledCase, Labels, Mlb

random.seed(42)


def review_pipe(datazip_path: str, review_path: str, return_xy=False):
    raw_data = io_utils.load_datazip(datazip_path)
    x = raw_data.x_dict
    y = raw_data.y_tags
    labelled_cases = [LabelledCase(k, v) for k, v in zip(x, y)]
    reviews = data_utils.load_labelled_cases(review_path)
    reviewed = edit_review(labelled_cases, reviews)
    new_x, new_y = data_utils.labelled_cases_to_xy(reviewed)
    if return_xy:
        return new_x, new_y
    zipname = data_utils.split_and_dump_dataset(new_x, new_y, train_first=True)
    return zipname


def enrich(
    model,
    mlb: Mlb,
    all_cases: List[Case],
    labelled_cases: List[LabelledCase],
    thresh=20,
    output_p=None,
):
    """Sample CRL cases from unlabelled cases from [all_cases]

    Args:
        model (StandaloneModel): a StandaloneModel
        mlb (Mlb): MultiLabelBinarizer
        all_cases (List[Case]): 11k raw cases
        labelled_cases (List[LabelledCase]): labelled cases
        thresh (int, optional): the CRL threshold. Defaults to 20.
        output_p (str, optional): the output path. Defaults to None.
    Returns:
        output_p (str): the path of a Json file with sampled cases

    """
    known_cases, known_tags = data_utils.unwrap_labelled_cases(labelled_cases)
    unlabelled_cases = data_utils.cases_minus(all_cases, known_cases)
    pred_tags: list = model.predict_tags(unlabelled_cases, mlb)
    needed = get_needed(known_tags, pred_tags, thresh=thresh)
    collection = collect(unlabelled_cases, needed, pred_tags)
    if output_p is None:
        output_p = f"enrich_{data_utils.get_timestamp()}.json"
    io_utils.dump_labelled_cases(collection, output_p)


def edit_review(old_cases: List[LabelledCase], reviews: List[LabelledCase]):
    cases_copy = old_cases.copy()
    review_map = {"".join(review.text.values()): review.tag for review in reviews}
    used_keys = set()
    for idx, case in enumerate(old_cases):
        key = "".join(case.text.values())
        cases_copy[idx].tag = data_utils.add_acute_LL(cases_copy[idx].tag)
        if key in review_map:
            cases_copy[idx].tag = review_map[key]
            used_keys.add(key)

    for review_idx, key in enumerate(review_map):
        if key not in used_keys:
            cases_copy.append(reviews[review_idx])
    return cases_copy


def get_needed(known_tags: Labels, pred_tags: Labels, thresh=20):
    def sampleable(tag, lib: Dict[str, list], need: dict):
        return len(lib.get(tag, [])) > need[tag]

    have = data_utils.count_tags(known_tags)
    need_num = {}
    for tag, num in have.items():
        if num < thresh:
            need_num[tag] = thresh - num
    lib = data_utils.grouping_idx(pred_tags)

    needed = {
        tag: random.sample(lib[tag], need_num[tag])
        for tag in need_num
        if sampleable(tag, lib, need_num)
    }
    return needed


def collect(unlabelled_cases, needed: dict, pred_tags: list):
    collection: List[LabelledCase] = []
    for indexes in needed.values():
        collection.extend(
            LabelledCase(unlabelled_cases[idx], pred_tags[idx]) for idx in indexes
        )
    return collection
