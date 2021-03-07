import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tagc.data_pipe import dataset_split, make_history_df, replay_ac
from tagc.data_utils import rawdata_stat
from tagc.io_utils import load_datazip
from tagc.visulisation import plot_tag_stat


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


def test_split():
    final_dsp = "out/stdDs.zip"

    acs = list(Path("data/labels").glob("*.json"))
    with TemporaryDirectory() as temp_dir:
        _, _, ac_dsps = replay_ac(acs, temp_dir)
        shutil.copy(ac_dsps[-1], final_dsp)

    dst = "out"
    dsps = dataset_split(final_dsp, dst)
    for idx, dsp in enumerate(dsps):
        rawdata = load_datazip(dsp)
        tag_stat = rawdata_stat(rawdata)
        tag_stat.to_csv(f"{dst}/{idx}data_stat.csv")
        fig = plot_tag_stat(tag_stat)
        fig.write_image(f"{dst}/{idx}data_stat.pdf")
