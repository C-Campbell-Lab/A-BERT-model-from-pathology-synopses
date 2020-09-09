from typing import List

from . import data_utils as du
from .domain import LabelledCase


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
    du.split_and_dump_dataset(new_x, new_y)
