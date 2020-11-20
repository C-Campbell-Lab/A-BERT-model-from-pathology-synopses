import gc
import shutil

import torch
from tagc.visulisation import plot_tag_performance
from tagc.validation import judge_on_tag
from tagc.domain import Params
from tagc.model import Classification, StandaloneModel
from tagc.train import Pipeline
from tagc.io_utils import dump_datazip, dump_json, load_datazip, load_json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from eval import form_eval
from tagc.evaluation import active_eval, form_pred

BEST_MODEL_P = "lab3/keepKey_200/model"


def add_training(
    eval_ret: str,
    base_path="dataset.zip",
    unlabelled_p="outputs/unlabelled.json",
    outdir="activeL",
    idx_marker=1,
):
    ds = load_datazip(base_path)
    df = pd.read_csv(eval_ret).drop_duplicates(subset=["ID", "Judge"], keep="last")
    indices = df["ID"].to_list()
    sampled_cases = load_json(unlabelled_p)
    add_texts = [sampled_cases[idx] for idx in indices]

    mlb = MultiLabelBinarizer()
    mlb.fit(ds.y_tags)
    y_true_ = df["eval"].map(lambda x: x.split(", ")).to_list()

    ds.x_train_dict = ds.x_train_dict + add_texts
    ds.y_train_tags = ds.y_train_tags + y_true_
    ds.x_dict = ds.x_dict + add_texts
    ds.y_tags = ds.y_tags + y_true_

    _ = dump_datazip(ds, f"{outdir}/dataset{idx_marker}.zip")
    return ds


def active_train(
    rawdata, outdir="activeL", over=5, epoch=10, idx_marker=1, init_model=BEST_MODEL_P
):
    params = Params(rawdata, 150, 200, 0.5, "bert-base-uncased", True, epoch)
    pipeline = Pipeline(params)
    pipeline.model = Classification.from_pretrained(init_model)
    model_p = f"{outdir}/model"
    pipeline.train(output_dir=model_p)
    pipeline.trainer.save_model(f"{outdir}/model{idx_marker}")
    standalone_model = StandaloneModel(pipeline.model, max_len=150, keep_key=True)
    form_eval(
        standalone_model,
        pipeline.mlb,
        outdir=outdir,
        marker=str(idx_marker),
        skip_state=True,
    )

    performance, metric, _ = judge_on_tag(
        standalone_model, pipeline.mlb, rawdata, n=over
    )
    dump_json(f"{outdir}/{over}_overall{idx_marker}.json", metric)
    performance.to_csv(f"{outdir}/{over}_Perf_tag{idx_marker}.csv")
    fig = plot_tag_performance(performance, metric, auc=True)
    fig.write_image(f"{outdir}/{over}_Perf_tag_auc{idx_marker}.pdf")
    fig = plot_tag_performance(performance, metric, auc=False)
    fig.write_image(f"{outdir}/{over}_Perf_tag_f1{idx_marker}.pdf")
    del pipeline
    del standalone_model
    shutil.rmtree(model_p)
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def cycle(
    eval_ret="cathy0.csv",
    base_path="dataset.zip",
    unlabelled_p="outputsK/unlabelled.json",
    outdir="activeL",
    eval_json="outputsX/eval.json",
    idx_marker=1,
):
    eval_over = active_eval(eval_ret, form_pred(eval_json))
    dump_json(f"{outdir}/eval_over{idx_marker}.json", eval_over)
    ds = add_training(
        eval_ret,
        base_path=base_path,
        unlabelled_p=unlabelled_p,
        idx_marker=idx_marker,
    )
    active_train(ds, outdir=outdir, idx_marker=idx_marker)
    eval_over = active_eval(eval_ret, form_pred(f"{outdir}/eval{idx_marker}.json"))
    dump_json(f"{outdir}/eval_over_after{idx_marker}.json", eval_over)


def main(
    eval_ret="cathy_j.csv",
    unlabelled_p="outputsK/unlabelled.json",
    outdir="activeL",
    batch_size=200,
):
    df = pd.read_csv(eval_ret).drop_duplicates(subset=["ID", "Judge"], keep="last")
    for idx, step in enumerate(range(0, len(df), batch_size), start=1):
        batch_df = df.iloc[step : step + batch_size]
        if len(batch_df) < batch_size:
            break
        batch_eval_p = f"{outdir}/cathy{idx}.csv"
        batch_df.to_csv(batch_eval_p, index=None)
        base_path = f"{outdir}/dataset{idx-1}.zip"
        eval_json = f"{outdir}/eval{idx-1}.json"
        cycle(batch_eval_p, base_path, unlabelled_p, outdir, eval_json, idx)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
