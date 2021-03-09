import gc
import os
import random
from tagc.make_figs import make_figures


import torch
from sklearn.preprocessing import MultiLabelBinarizer

from tagc.cal_thresh import analysis_kf
from tagc.domain import Params
from tagc.io_utils import (
    load_datazip,
    load_json,
)

from tagc.train import Pipeline

random.seed(42)


def train_main_model(rawdata, save=False, outdir="TagModelK"):
    params = Params(rawdata, 150, 200, 0.5, "bert-base-uncased", True, 10)
    pipeline = Pipeline(params)
    pipeline.train()
    example, judges_count, data, df = pipeline.validation_examples()
    print(judges_count)
    pipeline.trainer.save_model(outdir)
    if save:
        os.system(
            f"zip {outdir}.zip {outdir}/config.json {outdir}/pytorch_model.bin {outdir}/training_args.bin"
        )
    del pipeline
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def main(
    dataset_path: str,
    model_p="labF/keepKey_200/model/",
    case_p="case.json",
    plot=True,
    train=False,
    run_thresh=False,
    output_p="outputsS",
):
    ds = load_datazip(dataset_path)
    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    os.makedirs(f"{output_p}", exist_ok=True)

    if train:
        train_main_model(ds)
    if run_thresh:
        rets = load_json("cv_result.json")
        analysis_kf(rets, mlb, f"{output_p}/")
    if plot:
        make_figures(model_p, ds, case_p, dst=output_p)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
