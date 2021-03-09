import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tagc.data_pipe import (
    dataset_split,
    form_random_ds,
    make_history_df,
    replay_ac,
    sample_evaluation_from_path,
    xlsx_to_cases,
)
from tagc.data_utils import rawdata_stat
from tagc.io_utils import load_datazip, load_json
from tagc.visualization import plot_tag_stat

CASE_NUM = 11418


@pytest.fixture
def all_cases():
    _all_cases = load_json("data/cases.json")
    assert len(_all_cases) == CASE_NUM
    return _all_cases


@pytest.mark.skip
def test_xlsx_to_cases(all_cases):
    xlsx_p = "data/report.xlsx"
    cases = xlsx_to_cases(xlsx_p)
    assert len(cases) == CASE_NUM
    assert cases[0] == all_cases[0]
    assert cases[-1] == all_cases[-1]


def test_sample_evaluation():
    dsp = "out/standardDs.zip"
    cases_p = "data/cases.json"
    sampled_cases = sample_evaluation_from_path(cases_p, dsp)
    unlabelled = load_json("data/unlabelled.json")
    assert len(sampled_cases) == len(unlabelled)
    assert sampled_cases == unlabelled


@pytest.mark.skip
def test_replay_ac():
    acs = list(Path("data/labels").glob("*.json"))
    with TemporaryDirectory() as temp_dir:
        history, sizes, _ = replay_ac(acs, temp_dir)
    hist_df = make_history_df(history, sizes)
    assert hist_df["Sample Count"].to_list() == [
        50,
        83,
        198,
        282,
        296,
        344,
        393,
        485,
        500,
    ]
    assert hist_df["Label Count"].max() == 21


@pytest.fixture
def final_dsp():
    final_dsp_ = "out/standardDs.zip"
    acs = list(Path("data/labels").glob("*.json"))
    with TemporaryDirectory() as temp_dir:
        _, _, ac_dsps = replay_ac(acs, temp_dir)
        shutil.copy(ac_dsps[-1], final_dsp_)
    rawdata = load_datazip(final_dsp_)
    assert len(rawdata.y_test_tags) == 100
    return final_dsp_


@pytest.mark.skip
def test_split(final_dsp):
    dst = "out/standard"
    dsps = dataset_split(final_dsp, dst)
    for idx, dsp in enumerate(dsps):
        rawdata = load_datazip(dsp)
        tag_stat = rawdata_stat(rawdata)
        tag_stat.to_csv(f"{dst}/data_stat{idx}.csv")
        fig = plot_tag_stat(tag_stat)
        fig.write_image(f"{dst}/data_stat{idx}.pdf")
        rawdata = load_datazip(f"{dst}/standardDs{idx}.zip")
        assert len(rawdata.y_test_tags) == 100


@pytest.mark.skip
def test_random_ds():
    standard_dsps = [f"out/standard/standardDs{idx}.zip" for idx in range(3)]
    eval_ret = "data/evaluation/mona_j.csv"
    unlabelled_p = "data/unlabelled.json"
    dst = "out/random"
    dsps = form_random_ds(standard_dsps, eval_ret, unlabelled_p, outdir=dst)
    for idx, dsp in enumerate(dsps):
        rawdata = load_datazip(dsp)
        tag_stat = rawdata_stat(rawdata)
        tag_stat.to_csv(f"{dst}/random_data_stat{idx}.csv")
        fig = plot_tag_stat(tag_stat)
        fig.write_image(f"{dst}/random_data_stat{idx}.pdf")
        rawdata = load_datazip(f"{dst}/randomDs{idx}.zip")
        assert len(rawdata.y_test_tags) == 100
