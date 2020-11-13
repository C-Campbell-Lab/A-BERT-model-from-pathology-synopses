from tagc.visulisation import plot_tag_performance
from tagc.validation import judge_on_tag
from tagc.domain import Params
from tagc.model import Classification, StandaloneModel
from tagc.train import Pipeline
from tagc.io_utils import dump_datazip, dump_json, load_datazip, load_json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from eval import form_eval


def add_training(
    eval_ret: str,
    base_path="dataset.zip",
    unlabelled_p="outputs/unlabelled.json",
    outdir="activeL",
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

    _ = dump_datazip(ds, f"{outdir}/ACdataset.zip")
    return ds


def active_train(rawdata, outdir="activeL", over=5):
    model_p = "lab4/keepKey_200/model"
    params = Params(rawdata, 150, 200, 0.5, "bert-base-uncased", True, 5)
    pipeline = Pipeline(params)
    pipeline.model = Classification.from_pretrained(model_p)
    pipeline.train()
    pipeline.trainer.save_model(f"{outdir}/model")
    standalone_model = StandaloneModel(pipeline.model, max_len=150, keep_key=True)
    form_eval(standalone_model, pipeline.mlb, outdir=outdir)

    performance, metric, _ = judge_on_tag(
        standalone_model, pipeline.mlb, rawdata, n=over
    )
    dump_json(f"{outdir}/{over}_overall.json", metric)
    performance.to_csv(f"{outdir}/{over}_Perf_tag.csv")
    fig = plot_tag_performance(performance, metric, auc=True)
    fig.write_image(f"{outdir}/{over}_Perf_tag_auc.pdf")
    fig = plot_tag_performance(performance, metric, auc=False)
    fig.write_image(f"{outdir}/{over}_Perf_tag_f1.pdf")


def main(
    eval_ret="prediction judgement - Sheet1.csv",
    base_path="dataset.zip",
    unlabelled_p="outputsK/unlabelled.json",
    outdir="activeL",
):
    ds = add_training(eval_ret, base_path=base_path, unlabelled_p=unlabelled_p)
    active_train(ds, outdir=outdir)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
