import gc
import os
import random
from copy import copy

import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from tagc import data_utils
from tagc.cal_thresh import analysis_kf
from tagc.data_utils import count_tags, get_timestamp, rawdata_stat
from tagc.domain import Params
from tagc.io_utils import (
    build_eval_json,
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
    dimension_reduction_plot,
    get_tag_states,
    get_unlabelled_state,
    judge_on_num,
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


def train_main_model(rawdata):
    params = Params(rawdata, 100, 200, 0.5, "bert-base-uncased", False, 10)
    pipeline = Pipeline(params)
    pipeline.train()
    example, judges_count, data, df = pipeline.validation_examples()
    print(judges_count)
    df.to_csv(get_timestamp() + ".csv")
    pipeline.trainer.save_model("TagModel")
    os.system(
        "zip model.zip TagModel/config.json TagModel/pytorch_model.bin TagModel/training_args.bin"
    )
    del pipeline
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def make_figures(rawdata, mlb, adjust_thresh=False):
    model = StandaloneModel.from_path("TagModel", keep_key=False, max_len=100)

    # Rawdata_stat
    fn = "outputs/tag_stat.csv"
    if os.path.exists(fn):
        tag_stat = pd.read_csv(fn, index_col=0)
    else:
        tag_stat = rawdata_stat(rawdata)
        tag_stat.to_csv(fn)
    fig = plot_tag_stat(tag_stat)
    fig.update_layout(
        width=1280,
        height=600,
    )
    fig.write_image("outputs/data_stat.pdf")

    # Unlabelled
    fn = "outputs/unlabel_tsne.csv"
    if os.path.exists(fn):
        unstate_df = pd.read_csv(fn, index_col=0)
    else:
        unlabelled_p = "outputs/unlabelled.json"
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
            dump_json("outputs/unlabelled.json", sampled_cases)

        sampled_state = get_unlabelled_state(model, sampled_cases, mlb)
        dump_state(sampled_state, state_p="outputs/unstate.pkl")
        unstate_df = dimension_reduction(sampled_state, "TSNE", n_components=2)
        unstate_df.to_csv(fn)
        preds = model.over_predict(sampled_cases, n=5)
        thresh_items = label_output(preds)
        pred_prob = [list(zip(mlb.classes_, pred)) for pred in preds]
        eval_json = build_eval_json(sampled_cases, pred_prob, thresh_items)
        dump_json("outputs/eval.json", eval_json)
    fig = dimension_reduction_plot(unstate_df, n_components=2)
    fig.update_layout(
        width=1280,
        height=600,
    )
    fig.write_image("outputs/unlabelled_TSNE.pdf")

    # Labelled
    fn = "outputs/label_tsne.csv"
    if os.path.exists(fn):
        state_df = pd.read_csv(fn, index_col=0)
    else:
        states = get_tag_states(model, rawdata, mlb)
        state_df = dimension_reduction(states, "TSNE", n_components=2)
        state_df.to_csv(fn)
    fig = state_plot(state_df, 12)
    fig.write_image("outputs/fig3a_TSNE.pdf")

    # Summary
    example, j_count, data, df = summary(
        rawdata.x_test_dict,
        rawdata.y_test_tags,
        model.over_predict_tags(rawdata.x_test_dict, mlb, n=5),
    )
    df.to_csv("outputs/summary.csv")
    fig = plot_summary(data)
    fig.write_image("outputs/fig3b_Pie.pdf")
    performance_summary = judge_on_summary(df)
    fig = plot_num_performance(performance_summary)
    fig.write_image("outputs/fig3c_Perf_sum.pdf")
    review = []
    for case, pred_tag, true_tag, judge in example:
        if "Label" in judge:
            review.append({"text": case, "pred_tag": pred_tag, "tag": true_tag})
    dump_json("outputs/review.json", review)
    # Performance
    performance, overall = judge_on_tag(model, mlb, rawdata, n=5)
    performance.to_csv("outputs/Perf_tag.csv")
    fig = plot_tag_performance(performance, overall)
    fig.write_image("outputs/fig3c_Perf_tag.pdf")
    performance_n = judge_on_num(model, mlb, rawdata, n=5)
    performance_n.to_csv("outputs/Perf_num.csv")
    fig = plot_num_performance(performance_n)
    fig.write_image("outputs/fig3c_Perf_num.pdf")

    # Keyword
    fn = "outputs/top_key.json"
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
    fig.write_image("outputs/fig4_Kws.pdf")


def kf_flow(ds):
    kf = KFold(n_splits=5)
    cv_result = []
    for train, test in kf.split(ds.x_train_dict):
        tmp_ds = copy(ds)
        tmp_ds.x_train_dict = [ds.x_train_dict[idx] for idx in train]
        tmp_ds.y_train_tags = [ds.y_train_tags[idx] for idx in train]
        tmp_ds.x_test_dict = [ds.x_train_dict[idx] for idx in test]
        tmp_ds.y_test_tags = [ds.y_train_tags[idx] for idx in test]
        params = Params(tmp_ds, 100, 200, 0.5, "bert-base-uncased", False, 8)
        pipeline = Pipeline(params)
        pipeline.train()
        model = StandaloneModel(pipeline.model, pipeline.tokenizer)
        probs = model.predict_prob(tmp_ds.x_test_dict, pipeline.mlb)
        probs = [[(n, str(p)) for n, p in prob] for prob in probs]
        cv_result.append({"prob": probs, "tag": tmp_ds.y_test_tags})
        del model
        del pipeline
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
    dump_json("cv_result.json", cv_result)


def main(dataset_path, run_kf=False, plot=True, train=False, run_thresh=False):
    ds = load_datazip(dataset_path)
    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    os.makedirs("outputs", exist_ok=True)

    if train:
        train_main_model(ds)
    if run_kf:
        kf_flow(ds)
    if run_thresh:
        rets = load_json("cv_result.json")
        analysis_kf(rets, mlb, "outputs/")
    if plot:
        make_figures(ds, mlb)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
