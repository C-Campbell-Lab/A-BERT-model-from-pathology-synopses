from copy import copy
from data_size import eval_model
import os

from tagc.data_utils import rawdata_stat

from tagc.io_utils import load_datazip
from tagc.domain import Params, RawData

from tagc.model import StandaloneModel
from tagc.train import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
import gc
import torch
import shutil
from pathlib import Path


def train_eval(
    dsp,
    output_p,
    upsample=200,
    keep_key=True,
    over=5,
    cv=True,
):
    max_len = 150
    marker = Path(dsp).stem
    fn = f"{output_p}/{marker}tag_stat.csv"
    ds = load_datazip(dsp)
    tag_stat = rawdata_stat(ds)
    tag_stat.to_csv(fn)

    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    if not cv:
        single_train_eval(ds, max_len, upsample, keep_key, over, mlb, output_p, marker)
    else:
        for idx, tmp_ds in enumerate(kf_flow(ds)):
            single_train_eval(
                tmp_ds,
                max_len,
                upsample,
                keep_key,
                over,
                mlb,
                output_p,
                marker + f"cv{idx}",
            )


def single_train_eval(ds, max_len, upsample, keep_key, over, mlb, output_p, marker):
    params = Params(ds, max_len, upsample, 0.5, "bert-base-uncased", keep_key, 10, mlb)
    print(len(ds.x_train_dict))
    pipeline = Pipeline(params)
    model_p = pipeline.train(output_dir=output_p)
    model = StandaloneModel(
        pipeline.model, pipeline.tokenizer, keep_key=keep_key, max_len=max_len
    )
    eval_model(model, ds, over, mlb, output_p, marker)
    del pipeline
    shutil.rmtree(model_p)
    del model
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def run_replay(dsp_dir, outdir="replay"):
    dsps = Path(dsp_dir).glob("*.zip")
    os.makedirs(outdir, exist_ok=True)
    for dsp in dsps:
        # if os.path.exists(f"{outdir}/{Path(dsp).stem}_5_overall.json"):
        #     continue
        train_eval(dsp, outdir)


def kf_flow(ds: RawData):
    kf = KFold(n_splits=5)
    for train, test in kf.split(ds.x_dict):
        tmp_ds = copy(ds)
        tmp_ds.x_train_dict = [ds.x_dict[idx] for idx in train]
        tmp_ds.y_train_tags = [ds.y_tags[idx] for idx in train]
        tmp_ds.x_test_dict = [ds.x_dict[idx] for idx in test]
        tmp_ds.y_test_tags = [ds.y_tags[idx] for idx in test]
        yield tmp_ds


if __name__ == "__main__":
    import fire

    fire.Fire(run_replay)
