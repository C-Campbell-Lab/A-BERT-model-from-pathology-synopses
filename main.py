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
from tagc.data_utils import (
    count_tags,
    labelled_cases_to_xy,
    load_labelled_cases,
    rawdata_stat,
    split_and_dump_dataset,
)
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
    params = Params(rawdata, 150, 200, 0.5, "bert-base-uncased", True, 10)
    pipeline = Pipeline(params)
    pipeline.train()
    example, judges_count, data, df = pipeline.validation_examples()
    print(judges_count)
    pipeline.trainer.save_model("TagModelK")
    os.system(
        "zip modelK.zip TagModelK/config.json TagModelK/pytorch_model.bin TagModelK/training_args.bin"
    )
    del pipeline
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def make_figures(rawdata, mlb, output_p="outputs"):
    model = StandaloneModel.from_path(
        "lab4/keepKey_200/model", keep_key=True, max_len=150
    )
    over = 5
    # Rawdata_stat
    fn = f"{output_p}/tag_stat.csv"
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
    fig = dimension_reduction_plot(unstate_df, n_components=2)
    fig.update_layout(
        width=1280,
        height=600,
    )
    fig.write_image(f"{output_p}/unlabelled_TSNE.pdf")

    # Labelled
    fn = f"{output_p}/label_tsne.csv"
    if os.path.exists(fn):
        state_df = pd.read_csv(fn, index_col=0)
    else:
        states = get_tag_states(model, rawdata, mlb)
        state_df = dimension_reduction(states, "TSNE", n_components=2)
        state_df.to_csv(fn)
    fig = state_plot(state_df, 12)
    fig.write_image(f"{output_p}/fig3a_TSNE.pdf")

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
    fig.write_image(f"{output_p}/fig4_Kws.pdf")


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


def main(
    dataset_path: str,
    run_kf=False,
    plot=True,
    train=False,
    run_thresh=False,
    output_p="outputsK",
):
    if dataset_path.endswith(".json"):
        json_data = load_labelled_cases(dataset_path)
        dataset_path = split_and_dump_dataset(*labelled_cases_to_xy(json_data))
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
