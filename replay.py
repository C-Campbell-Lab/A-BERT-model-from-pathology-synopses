from data_size import eval_model
import os

from tagc.data_utils import rawdata_stat

from tagc.io_utils import load_datazip
from tagc.domain import Params

from tagc.model import StandaloneModel
from tagc.train import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
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
):
    max_len = 150
    marker = Path(dsp).stem
    fn = f"{output_p}/{marker}tag_stat.csv"
    ds = load_datazip(dsp)
    tag_stat = rawdata_stat(ds)
    tag_stat.to_csv(fn)

    mlb = MultiLabelBinarizer().fit(ds.y_tags)
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
        if os.path.exists(f"{outdir}/{Path(dsp).stem}_5_overall.json"):
            continue
        train_eval(dsp, outdir)


if __name__ == "__main__":
    import fire

    fire.Fire(run_replay)
