import random
from typing import List

from . import data_utils as du
from .domain import LabelledCase, Mlb, Tags
from .model import StandaloneModel

random.seed(42)


def edit_review(old_cases: List[LabelledCase], reviews: List[LabelledCase]):
    cases_copy = old_cases.copy()
    review_map = {"".join(review.text.values()): review.tag for review in reviews}
    for idx, case in enumerate(old_cases):
        key = "".join(case.text.values())
        cases_copy[idx].tag = du.add_acute_LL(cases_copy[idx].tag)
        if key in review_map:
            cases_copy[idx].tag = review_map[key]
    return cases_copy


def review_pipe(datazip_path: str, review_path: str):
    raw_data = du.load_datazip(datazip_path)
    x = raw_data.x_dict
    y = raw_data.y_tags
    labelled_cases = [LabelledCase(k, v) for k, v in zip(x, y)]
    reviews = du.load_labelled_cases(review_path)
    reviewed = edit_review(labelled_cases, reviews)
    new_x, new_y = du.labelled_cases_to_xy(reviewed)
    zipname = du.split_and_dump_dataset(new_x, new_y)
    return zipname


def enrich(
    model: StandaloneModel, mlb: Mlb, all_cases, labelled_cases: List[LabelledCase]
):
    known_cases, known_tags = du.unwrap_labelled_cases(labelled_cases)
    unlabelled_cases = du.cases_minus(all_cases, known_cases)
    preds = model.predict(unlabelled_cases)
    pred_tags = mlb.inverse_transform(preds > 0.5)
    needed = get_needed(known_tags, pred_tags)
    collection = collect(unlabelled_cases, needed, pred_tags)
    du.dump_labelled_cases(collection, f"enrich_{du.get_timestamp()}.json")


def get_needed(known_tags: Tags, pred_tags: Tags, thresh=20):
    def sampleable(tag, lib: dict, need: dict):
        return lib.get(tag, 0) > need[tag]

    have = du.count_tags(known_tags)
    need_num = {}
    for tag, num in have.items():
        if num < thresh:
            need_num[tag] = thresh - num
    lib = du.grouping_idx(pred_tags)

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
