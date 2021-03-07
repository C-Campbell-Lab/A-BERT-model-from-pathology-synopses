import random
from itertools import accumulate
from typing import List

import pandas as pd

from tagc.data_utils import (
    count_tags,
    labelled_cases_to_xy,
    load_labelled_cases,
    split_and_dump_dataset,
    train_test_split,
)
from tagc.domain import RawData
from tagc.io_utils import dump_datazip, load_datazip, load_json
from tagc.review import review_pipe

random.seed(42)


def clean(x_dict, y_tags):
    content = sorted(
        (("".join(v.values()), idx) for idx, v in enumerate(x_dict)),
        key=lambda x: x[0],
    )
    for c1, c2 in zip(content, content[1:]):  # check duplication
        if c1[0] == c2[0]:
            print(f"remove duplication, idx:{c1[1]}")
            del_idx = c1[1]
            x_dict.pop(del_idx)
            y_tags.pop(del_idx)
    for idx, tag in enumerate(y_tags):
        if "normal" in tag and "iron deficiency" not in tag:
            y_tags[idx] = ["normal"]
    x = x_dict
    y = y_tags
    x_train_dict, x_test_dict, y_train_tags, y_test_tags = train_test_split(
        x, y, test_size=0.2, train_first=True
    )
    rd = RawData(x, y, x_train_dict, y_train_tags, x_test_dict, y_test_tags)
    return rd


def replay_ac(ac_data_ps: List[str], dst="."):
    """Replay the active learning data sampling results

    Args:
        ac_data_ps (List[str]): File paths of labels

    Returns:
        history (List[Counter]): label count for each iteration
        sizes (List[int]): case number for each iteration
        dsps (List[str]): File paths of datasets
    """
    history = []
    sizes = []
    dsps = []
    start = load_labelled_cases(ac_data_ps[0])
    ds = clean(*labelled_cases_to_xy(start))
    dsp = dump_datazip(ds, f"{dst}/data0.zip")
    sizes.append(len(ds.x_dict))
    history.append(count_tags(ds.y_tags).keys())
    for idx, target in enumerate(ac_data_ps[1:], 1):
        ds = clean(*review_pipe(dsp, target, return_xy=True))
        dsp = dump_datazip(ds, f"{dst}/data{idx}.zip")
        dsps.append(dsp)
        sizes.append(len(ds.x_dict))
        history.append(count_tags(ds.y_tags).keys())
    return history, sizes, dsps


def make_history_df(history, sizes):
    diffs = []
    tag_count = []
    for i in range(len(history) - 1, 0, -1):
        diff = sorted(set(history[i]) - set(history[i - 1]))
        diffs.append(diff)
        tag_count.append(len(diff))
    diff = sorted(history[0])
    diffs.append(diff)
    tag_count.append(len(diff))
    tag_count = accumulate(reversed(tag_count))
    hist_df = pd.DataFrame(
        {
            "Iteration": list(range(1, len(sizes) + 1)),
            "New labels": [", ".join(item) for item in reversed(diffs)],
            "Label Count": list(tag_count),
            "Sample Count": sizes,
        }
    )
    return hist_df


def dataset_split(final_dsp, dst="."):
    ds = load_datazip(final_dsp)
    std_dsps = []
    for i in range(3):
        output = f"{dst}/stdDs{i}.zip"
        split_and_dump_dataset(ds.x_dict, ds.y_tags, train_first=False, output=output)
        std_dsps.append(output)
    return std_dsps


def form_random_ds(
    std_dsps: List[str],
    eval_ret="mona_j.csv",
    unlabelled_p="unlabelled.json",
    outdir=".",
):
    df = pd.read_csv(eval_ret).drop_duplicates(subset=["ID", "Judge"], keep="last")
    indices = df["ID"].to_list()
    sampled_cases = load_json(unlabelled_p)
    add_text = [sampled_cases[idx] for idx in indices]
    y_true_ = df["eval"].map(lambda x: x.split(", ")).to_list()
    for i, base_path in enumerate(std_dsps):
        ds = load_datazip(base_path)
        random_idx = random.sample(list(range(len(indices))), 400)
        x_train_dict = [add_text[idx] for idx in random_idx]
        y_train_tags = [y_true_[idx] for idx in random_idx]
        ds.x_train_dict = x_train_dict
        ds.y_train_tags = y_train_tags
        ds.x_dict = x_train_dict + ds.x_test_dict
        ds.y_tags = y_train_tags + ds.y_test_tags

        dsp = dump_datazip(ds, f"{outdir}/stdRandom{i}.zip")
        print(dsp)
