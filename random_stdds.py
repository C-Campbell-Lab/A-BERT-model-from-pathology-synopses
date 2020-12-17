import random

import pandas as pd

from tagc.io_utils import dump_datazip, load_datazip, load_json

random.seed(42)


def form_random_ds(
    base_path="stdDs.zip",
    eval_ret="mona_j.csv",
    unlabelled_p="unlabelled.json",
    outdir=".",
):
    ds = load_datazip(base_path)
    df = pd.read_csv(eval_ret).drop_duplicates(subset=["ID", "Judge"], keep="last")
    indices = df["ID"].to_list()
    sampled_cases = load_json(unlabelled_p)
    add_text = [sampled_cases[idx] for idx in indices]
    y_true_ = df["eval"].map(lambda x: x.split(", ")).to_list()
    for i in range(3):
        random_idx = random.sample(list(range(len(indices))), 400)
        x_train_dict = [add_text[idx] for idx in random_idx]
        y_train_tags = [y_true_[idx] for idx in random_idx]
        ds.x_train_dict = x_train_dict
        ds.y_train_tags = y_train_tags
        ds.x_dict = x_train_dict + ds.x_test_dict
        ds.y_tags = y_train_tags + ds.y_test_tags

        dsp = dump_datazip(ds, f"{outdir}/stdRandom{i}.zip")
        print(dsp)


if __name__ == "__main__":
    form_random_ds()
