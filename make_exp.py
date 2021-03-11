import gc
import os
import shutil
from copy import copy

import torch
from sklearn.preprocessing import MultiLabelBinarizer

from tagc.domain import Params
from tagc.io_utils import load_datazip
from tagc.model import StandaloneModel
from tagc.train import Pipeline
from tagc.validation import eval_model


def run_experiment(
    output_p="output",
    dataset_path="stdDs.zip",
    upsample=200,
    keep_key=True,
    over=None,
    step=50,
    metrics_only=False,
):
    if over is None:
        over = -1 if upsample == -1 else 5
    output_p = f"{output_p}/{'keepKey_' if keep_key else 'noKey_'}{upsample}"
    os.makedirs(output_p, exist_ok=True)
    ds = load_datazip(dataset_path)
    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    max_len = 150

    for size in range(step, len(ds.x_train_dict) + 1, step):
        if metrics_only:
            model = StandaloneModel.from_path(
                f"{output_p}/model", keep_key=keep_key, max_len=max_len
            )
            eval_model(model, ds, over, mlb, output_p, size)
        else:
            part_ds = slice_dataset(ds, size)

            params = Params(
                part_ds, max_len, upsample, 0.5, "bert-base-uncased", keep_key, 10, mlb
            )
            print(len(part_ds.x_train_dict))
            pipeline = Pipeline(params)
            model_p = pipeline.train(output_dir=output_p)
            model = StandaloneModel(
                pipeline.model, pipeline.tokenizer, keep_key=keep_key, max_len=max_len
            )
            eval_model(model, part_ds, over, mlb, output_p, size)
            if size >= len(ds.x_train_dict):
                pipeline.trainer.save_model(f"{output_p}/model")
            del pipeline
            shutil.rmtree(model_p)
        del model
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


def slice_dataset(ds, size):
    tmp_ds = copy(ds)
    tmp_ds.x_train_dict = ds.x_train_dict[:size]
    tmp_ds.y_train_tags = ds.y_train_tags[:size]
    return tmp_ds


if __name__ == "__main__":
    import fire

    fire.Fire(run_experiment)
