import gc
import os
import random
from copy import copy
from os.path import join

import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer

from data_size import size_effect
from tagc import data_utils
from tagc.cal_thresh import analysis_kf
from tagc.data_utils import count_tags, rawdata_stat
from tagc.domain import Params, RawData
from tagc.io_utils import (
    build_eval_json,
    dump_datazip,
    dump_json,
    dump_state,
    load_datazip,
    load_json,
)
from tagc.mask_explain import MaskExplainer, top_keywords
from tagc.model import StandaloneModel, label_output
from tagc.train import Pipeline
from tagc.validation import (
    dimension_reduction,
    get_tag_states,
    get_unlabelled_state,
    judge_on_summary,
    judge_on_tag,
    summary,
)
from tagc.visulisation import (
    kw_plot,
    plot_num_performance,
    plot_summary,
    plot_tag_performance,
    plot_tag_stat,
    state_plot,
)

random.seed(42)

BEST_MODEL_P = "labF/keepKey_200/model/"


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


def make_figures(rawdata, mlb, output_p="outputsT"):
    model = StandaloneModel.from_path(BEST_MODEL_P, keep_key=True, max_len=150)
    over = 5
    # Rawdata_stat
    fn = f"{output_p}/tag_stat.csv"
    if os.path.exists(fn):
        tag_stat = pd.read_csv(fn, index_col=0)
    else:
        tag_stat = rawdata_stat(rawdata)
        tag_stat.to_csv(fn)
    fig = plot_tag_stat(tag_stat)
    fig.write_image(f"{output_p}/data_stat.pdf")

    # Unlabelled
    fn = f"{output_p}/unlabel_tsne.csv"
    if os.path.exists(fn):
        unstate_df = pd.read_csv(fn, index_col=0)
    else:
        unlabelled_p = f"{output_p}/unlabelled.json"
        if os.path.exists(unlabelled_p):
            sampled_cases = load_json(unlabelled_p)
        else:
            all_cases = load_json("cases.json")
            known_cases, known_tags = data_utils.unwrap_labelled_cases(
                rawdata.to_labelled_cases()
            )
            unlabelled_cases = data_utils.cases_minus(all_cases, known_cases)
            k = 1000
            sampled_cases = random.sample(unlabelled_cases, k)
            dump_json(f"{output_p}/unlabelled.json", sampled_cases)

        sampled_state = get_unlabelled_state(model, sampled_cases, mlb)
        dump_state(sampled_state, state_p=f"{output_p}/unstate.pkl")
        unstate_df = dimension_reduction(sampled_state, "TSNE", n_components=2)
        unstate_df.to_csv(fn)
        preds = model.over_predict(sampled_cases, n=over)
        thresh_items = label_output(preds)
        pred_prob = [list(zip(mlb.classes_, pred)) for pred in preds]
        eval_json = build_eval_json(sampled_cases, pred_prob, thresh_items)
        dump_json(f"{output_p}/eval.json", eval_json)
    fig = state_plot(unstate_df, 12)
    fig.write_image(f"{output_p}/unlabelled_TSNE.pdf")
    fig.write_html(f"{output_p}/unlabel_tsne.html")

    # Labelled
    fn = f"{output_p}/label_tsne.csv"
    if os.path.exists(fn):
        state_df = pd.read_csv(fn, index_col=0)
    else:
        states = get_tag_states(model, rawdata, mlb)
        state_df = dimension_reduction(states, "TSNE", n_components=2)
        state_df.to_csv(fn)
    fig = state_plot(state_df, 12)
    fig.write_image(f"{output_p}/dev_tsne.pdf")
    fig.write_html(f"{output_p}/label_tsne.html")

    # Performance
    performance, metric, pred_tags = judge_on_tag(model, mlb, rawdata, n=over)
    dump_json(f"{output_p}/{over}_overall.json", metric)
    performance.to_csv(f"{output_p}/{over}_Perf_tag.csv")
    fig = plot_tag_performance(performance, metric, auc=True)
    fig.write_image(f"{output_p}/{over}_Perf_tag_auc.pdf")
    fig = plot_tag_performance(performance, metric, auc=False)
    fig.write_image(f"{output_p}/{over}_Perf_tag_f1.pdf")

    # Summary
    example, j_count, data, df = summary(
        rawdata.x_test_dict,
        rawdata.y_test_tags,
        pred_tags,
    )

    df.to_csv(f"{output_p}/summary.csv")
    fig = plot_summary(data)
    fig.write_image(f"{output_p}/fig3b_Pie.pdf")
    performance_summary = judge_on_summary(df)
    fig = plot_num_performance(performance_summary)
    fig.write_image(f"{output_p}/fig3c_Perf_sum.pdf")
    review = []
    for case, pred_tag, true_tag, judge in example:
        if "Label" in judge:
            review.append({"text": case, "pred_tag": pred_tag, "tag": true_tag})
    dump_json(f"{output_p}/review.json", review)

    fn = f"{output_p}/top_key.json"
    if os.path.exists(fn):
        top_key = load_json(fn)
    else:
        maskExplainer = MaskExplainer(mlb)
        top = top_keywords(maskExplainer, model, rawdata.x_dict)
        tag_counter = count_tags(rawdata.y_tags)
        large_enough = [k for k, v in tag_counter.items() if v >= 20]
        top_key = {}
        for t, v in top.items():
            if t in large_enough:
                top_key[t] = v
        dump_json(fn, top_key)
    fig = kw_plot(top_key)
    fig.write_image(f"{output_p}/knockout_result.pdf")


def kf_flow(ds: RawData, kf_out="kf_out"):
    kf = KFold(n_splits=5)
    for idx, (train, test) in enumerate(kf.split(ds.x_dict)):
        tmp_ds = copy(ds)
        tmp_ds.x_train_dict = [ds.x_dict[idx] for idx in train]
        tmp_ds.y_train_tags = [ds.y_tags[idx] for idx in train]
        tmp_ds.x_test_dict = [ds.x_dict[idx] for idx in test]
        tmp_ds.y_test_tags = [ds.y_tags[idx] for idx in test]
        output_p = f"{kf_out}/{idx}/"
        step = len(train)
        os.makedirs(output_p, exist_ok=True)
        dsp = dump_datazip(tmp_ds, join(output_p, f"ds{idx}.zip"))
        for upsample in (200, -200):
            for keep_key in (True, False):
                size_effect(
                    output_p=output_p,
                    dataset_path=dsp,
                    upsample=upsample,
                    keep_key=keep_key,
                    over=5,
                    step=step,
                    metrics_only=False,
                )


def main(
    dataset_path: str,
    run_kf=False,
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
    if run_kf:
        kf_flow(ds)
    if run_thresh:
        rets = load_json("cv_result.json")
        analysis_kf(rets, mlb, f"{output_p}/")
    if plot:
        make_figures(ds, mlb, output_p=output_p)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
