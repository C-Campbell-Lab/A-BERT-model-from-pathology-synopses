import os
from copy import copy
from tagc.io_utils import load_datazip, dump_json
from tagc.domain import Params
from tagc.validation import judge_on_tag, summary
from tagc.model import StandaloneModel
from tagc.train import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from tagc.visulisation import plot_tag_performance
import gc
import torch
import shutil


def run_exp(output_p, metrics_only=True, over=5):
    for upsample in (200, -200):
        for keep_key in (True, False):
            size_effect(
                output_p=output_p,
                dataset_path="dataset.zip",
                upsample=upsample,
                keep_key=keep_key,
                over=over,
                step=400,
                metrics_only=metrics_only,
            )


def size_effect(
    output_p="up_effect",
    dataset_path="dataset.zip",
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
            eval_model(model, ds, over, mlb, output_p, size)
            if size >= len(ds.x_train_dict):
                pipeline.trainer.save_model(f"{output_p}/model")
            del pipeline
            shutil.rmtree(model_p)
        del model
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


def eval_model(model, ds, over, mlb, output_p, size):

    performance, metric, pred_tags = judge_on_tag(model, mlb, ds, n=over)
    dump_json(f"{output_p}/{size}_{over}_overall.json", metric)
    performance.to_csv(f"{output_p}/{size}_{over}_Perf_tag.csv")
    fig = plot_tag_performance(performance, metric)
    fig.write_image(f"{output_p}/{size}_{over}_Perf_tag.pdf")

    _, _, _, df = summary(
        ds.x_test_dict,
        ds.y_test_tags,
        pred_tags,
    )
    df.to_csv(f"{output_p}/{size}_{over}_summary.csv")


def slice_dataset(ds, size):
    tmp_ds = copy(ds)
    tmp_ds.x_train_dict = ds.x_train_dict[:size]
    tmp_ds.y_train_tags = ds.y_train_tags[:size]
    return tmp_ds


if __name__ == "__main__":
    import fire

    fire.Fire(run_exp)
