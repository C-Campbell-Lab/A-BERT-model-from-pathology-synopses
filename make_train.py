import gc
import random
import shutil

import torch
from sklearn.preprocessing import MultiLabelBinarizer

from tagc.domain import Params, RawData
from tagc.io_utils import load_datazip
from tagc.make_figs import make_figures
from tagc.model import StandaloneModel
from tagc.train import Pipeline
from tagc.validation import eval_model

random.seed(42)


def train_main_model(
    dataset: RawData, model_p="model", outdir="out", epoch=10, upsmaple=200
):
    keep_key = True
    max_len = 150
    dropout = 0.5
    mlb = MultiLabelBinarizer().fit(dataset.y_tags)
    params = Params(
        dataset, max_len, upsmaple, dropout, "bert-base-uncased", keep_key, epoch
    )
    pipeline = Pipeline(params)
    model_tmp = pipeline.train(output_dir=outdir)
    model = StandaloneModel(
        pipeline.model, pipeline.tokenizer, keep_key=keep_key, max_len=max_len
    )
    _, judges_count, _, _ = eval_model(model, dataset, 5, mlb, outdir)
    print(judges_count)
    pipeline.trainer.save_model(model_p)
    del pipeline
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    shutil.rmtree(model_tmp)


def main(
    dataset_p: str,
    unlabelled_p: str,
    model_p: str,
    outdir: str,
    plot=True,
    train=False,
):
    if train:
        ds = load_datazip(dataset_p)
        train_main_model(ds, model_p=model_p)
    if plot:
        make_figures(model_p, dataset_p, unlabelled_p, dst=outdir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
