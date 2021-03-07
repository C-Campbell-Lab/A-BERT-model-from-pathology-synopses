from typing import List
from itertools import accumulate
import pandas as pd
from tagc.review import review_pipe
from tagc.io_utils import dump_datazip
from tagc.domain import RawData
from tagc.data_utils import (
    count_tags,
    labelled_cases_to_xy,
    load_labelled_cases,
    train_test_split,
)


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
        x, y, test_size=0.2
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
    """
    history = []
    sizes = []
    start = load_labelled_cases(ac_data_ps[0])
    ds = clean(*labelled_cases_to_xy(start))
    dsp = dump_datazip(ds, f"{dst}/data0.zip")
    sizes.append(len(ds.x_dict))
    history.append(count_tags(ds.y_tags).keys())
    for idx, target in enumerate(ac_data_ps[1:], 1):
        ds = clean(*review_pipe(dsp, target, return_xy=True))
        dsp = dump_datazip(ds, f"{dst}/data{idx}.zip")
        sizes.append(len(ds.x_dict))
        history.append(count_tags(ds.y_tags).keys())
    return history, sizes


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
