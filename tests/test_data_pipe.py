from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tagc.data_pipe import make_history_df, replay_ac, xlsx_to_cases
from tagc.io_utils import load_datazip, load_json
from tagc.make_ds import mk_randomDs, mk_standardDs, mk_unlabelled

CASE_NUM = 11418


@pytest.fixture
def all_cases():
    _all_cases = load_json("data/cases.json")
    assert len(_all_cases) == CASE_NUM
    return _all_cases


def test_final_ds():
    acs = list(Path("data/labels").glob("*.json"))
    final_dsp = "out/standardDsTmp.zip"
    with TemporaryDirectory() as temp_dir:
        _, _, ac_dsps = replay_ac(acs, temp_dir)
        rawdata = load_datazip(ac_dsps[-1])
        rawdata_ref = load_datazip(final_dsp)
        assert rawdata.y_test_tags == rawdata_ref.y_test_tags


@pytest.mark.skip
def test_xlsx_to_cases(all_cases):
    xlsx_p = "data/report.xlsx"
    cases = xlsx_to_cases(xlsx_p)
    assert len(cases) == CASE_NUM
    assert cases[0] == all_cases[0]
    assert cases[-1] == all_cases[-1]


@pytest.mark.skip
def test_sample_evaluation():
    final_dsp = "out/standardDs.zip"
    cases_p = "data/cases.json"
    with TemporaryDirectory() as temp_dir:
        unp = mk_unlabelled(final_dsp, cases_p, f"{temp_dir}/unl.json")
        sampled_cases = load_json(unp)
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


def test_split_ref():
    final_dsp = "out/standardDsTmp.zip"
    src = "out/standard"
    with TemporaryDirectory() as temp_dir:
        dsps = mk_standardDs(final_dsp, temp_dir, plot=False)
        for idx, dsp in enumerate(dsps):
            rawdata = load_datazip(dsp)
            ref_rawdata = load_datazip(f"{src}/standardDs{idx}.zip")
            assert rawdata.y_test_tags == ref_rawdata.y_test_tags


def test_random_ds():
    standard_dsps = [f"out/standard/standardDs{idx}.zip" for idx in range(3)]
    eval_ret = "data/evaluation/mona_j.csv"
    unlabelled_p = "data/unlabelled.json"
    src = "out/random"
    with TemporaryDirectory() as temp_dir:
        dsps = mk_randomDs(
            standard_dsps,
            eval_ret=eval_ret,
            unlabelled_p=unlabelled_p,
            dst=temp_dir,
            plot=False,
        )
        for idx, dsp in enumerate(dsps):
            rawdata = load_datazip(dsp)
            ref_rawdata = load_datazip(f"{src}/randomDs{idx}.zip")
            assert rawdata.y_test_tags == ref_rawdata.y_test_tags
