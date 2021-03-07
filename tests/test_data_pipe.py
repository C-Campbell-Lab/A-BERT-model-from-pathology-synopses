from tagc.data_pipe import make_history_df, replay_ac
from tempfile import TemporaryDirectory
from pathlib import Path


def test_replay_ac():
    acs = list(Path("data/labels").glob("*.json"))
    with TemporaryDirectory() as temp_dir:
        history, sizes = replay_ac(acs, temp_dir)
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
