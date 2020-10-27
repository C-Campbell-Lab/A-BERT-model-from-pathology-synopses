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


def pipe():
    size_effect(
        output_p="up_effect",
        dataset_path="dataset.zip",
        upsample=200,
        keep_key=False,
    )

    size_effect(
        output_p="up_effect",
        dataset_path="dataset.zip",
        upsample=-1,
        keep_key=False,
    )

    size_effect(
        output_p="up_effect",
        dataset_path="dataset.zip",
        upsample=200,
        keep_key=True,
    )

    size_effect(
        output_p="up_effect",
        dataset_path="dataset.zip",
        upsample=-1,
        keep_key=True,
    )


def size_effect(
    output_p="up_effect",
    dataset_path="dataset.zip",
    upsample=200,
    keep_key=False,
):
    over = -1 if upsample == -1 else 5
    output_p = f"{'keepKey_' if keep_key else ''}{output_p}{upsample if upsample != -1 else ''}"
    os.makedirs(output_p, exist_ok=True)
    ds = load_datazip(dataset_path)
    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    step = 50
    for size in range(step, len(ds.x_dict), step):
        part_ds = slice_dataset(ds, size)
        max_len = 150
        params = Params(
            part_ds, max_len, upsample, 0.5, "bert-base-uncased", keep_key, 10, mlb
        )
        print(len(part_ds.x_train_dict))
        pipeline = Pipeline(params)
        model_p = pipeline.train(output_dir=output_p)
        model = StandaloneModel(pipeline.model, pipeline.tokenizer, keep_key=keep_key)
        _, _, _, df = summary(
            ds.x_test_dict,
            ds.y_test_tags,
            model.over_predict_tags(ds.x_test_dict, mlb, n=over),
        )
        df.to_csv(f"{output_p}/{size}_summary.csv")
        performance, overall = judge_on_tag(model, mlb, ds, n=over)
        dump_json(
            f"{output_p}/{size}_overall.json", {k: str(v) for k, v in overall.items()}
        )
        performance.to_csv(f"{output_p}/{size}_Perf_tag.csv")
        fig = plot_tag_performance(performance, overall)
        fig.write_image(f"{output_p}/{size}_Perf_tag.pdf")
        del model
        del pipeline
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        shutil.rmtree(model_p)


def slice_dataset(ds, size):
    tmp_ds = copy(ds)
    tmp_ds.x_train_dict = ds.x_train_dict[:size]
    tmp_ds.y_train_tags = ds.y_train_tags[:size]
    return tmp_ds


if __name__ == "__main__":
    import fire

    fire.Fire(size_effect)
