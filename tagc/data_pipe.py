import random
from itertools import accumulate
from typing import List

import pandas as pd

from tagc.cleaner import Cleaner, has_content
from tagc.data_utils import (
    cases_minus,
    count_tags,
    labelled_cases_to_xy,
    load_labelled_cases,
    split_and_dump_dataset,
    train_test_split,
    unwrap_labelled_cases,
)
from tagc.domain import RawData
from tagc.io_utils import dump_datazip, load_datazip, load_json
from tagc.review import review_pipe

random.seed(42)


def xlsx_to_cases(xlsx_p):
    raw_data = pd.read_excel(xlsx_p, engine="openpyxl")
    need_fileted = raw_data.loc[
        ~(raw_data["BM Aspirate Results"].isna() | raw_data["BM ASP Number"].isna())
    ]
    need_fileted = need_fileted.loc[
        need_fileted["BMB Number"].str.contains("H")
        & need_fileted["BM ASP Number"].str.strip().str.endswith("R")
    ]
    raw_sents = []
    for row in need_fileted.iterrows():
        text = row[1]["BM Aspirate Results"]
        raw_sents.append(text)
    cleaner = Cleaner()
    cases = list(filter(has_content, map(cleaner.parse_aspirate_result, raw_sents)))
    return cases


def sample_evaluation_from_path(cases_p, dsp):
    all_cases = load_json(cases_p)
    dataset = load_datazip(dsp)
    return sample_evaluation(all_cases, dataset)


def sample_evaluation(cases, dataset, k=1000):
    known_cases, _ = unwrap_labelled_cases(dataset.to_labelled_cases())
    unlabelled_cases = cases_minus(cases, known_cases)
    sampled_cases = random.sample(unlabelled_cases, k)
    return sampled_cases


def sanity_check(x_dict, y_tags):
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
    ds = sanity_check(*labelled_cases_to_xy(start))
    dsp = dump_datazip(ds, f"{dst}/data0.zip")
    sizes.append(len(ds.x_dict))
    history.append(count_tags(ds.y_tags).keys())
    for idx, target in enumerate(ac_data_ps[1:], 1):
        ds = sanity_check(*review_pipe(dsp, target, return_xy=True))
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
    standard_dsps = []
    for i in range(3):
        output = f"{dst}/standardDs{i}.zip"
        split_and_dump_dataset(ds.x_dict, ds.y_tags, train_first=False, output=output)
        standard_dsps.append(output)
    return standard_dsps


def form_random_ds(
    standard_dsps: List[str],
    eval_ret="mona_j.csv",
    unlabelled_p="unlabelled.json",
    outdir=".",
    train_size=400,
):
    df = pd.read_csv(eval_ret).drop_duplicates(subset=["ID", "Judge"], keep="last")
    indices = df["ID"].to_list()
    sampled_cases = load_json(unlabelled_p)
    add_text = [sampled_cases[idx] for idx in indices]
    y_true_ = df["eval"].map(lambda x: x.split(", ")).to_list()
    dsps = []
    for i, base_path in enumerate(standard_dsps):
        ds = load_datazip(base_path)
        random_idx = random.sample(list(range(len(indices))), train_size)
        x_train_dict = [add_text[idx] for idx in random_idx]
        y_train_tags = [y_true_[idx] for idx in random_idx]
        ds.x_train_dict = x_train_dict
        ds.y_train_tags = y_train_tags
        ds.x_dict = x_train_dict + ds.x_test_dict
        ds.y_tags = y_train_tags + ds.y_test_tags

        dsp = dump_datazip(ds, f"{outdir}/randomDs{i}.zip")
        dsps.append(dsp)
        print(dsp)
    return dsps
