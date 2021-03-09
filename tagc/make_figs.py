import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from tagc.data_utils import rawdata_stat
from tagc.io_utils import (
    build_eval_json,
    dump_json,
    dump_state,
    load_datazip,
    load_json,
)
from tagc.mask_explain import MaskExplainer, top_keywords, filter_top
from tagc.model import StandaloneModel, label_output
from tagc.validation import (
    dimension_reduction,
    eval_model,
    get_tag_states,
    get_unlabelled_state,
)
from tagc.visualization import (
    kw_plot,
    plot_tag_stat,
    state_plot,
)
from tagc.data_pipe import sample_evaluation_from_path


def make_figures(model_p, dsp, case_p, dst="figF"):
    dataset = load_datazip(dsp)
    model = StandaloneModel.from_path(model_p, keep_key=True, max_len=150)
    over = 5
    mlb = MultiLabelBinarizer().fit(dataset.y_tags)
    os.makedirs(dst, exist_ok=True)
    # Rawdata_stat
    fn = f"{dst}/data_stat.csv"
    if os.path.exists(fn):
        tag_stat = pd.read_csv(fn, index_col=0)
    else:
        tag_stat = rawdata_stat(dataset)
        tag_stat.to_csv(fn)
    fig = plot_tag_stat(tag_stat)
    fig.write_image(f"{dst}/data_stat.pdf")

    # Unlabelled
    fn = f"{dst}/unlabel_tsne.csv"
    if os.path.exists(fn):
        unstate_df = pd.read_csv(fn, index_col=0)
    else:
        sampled_cases = sample_evaluation_from_path(dsp, case_p)
        sampled_state = get_unlabelled_state(model, sampled_cases, mlb)
        dump_state(sampled_state, state_p=f"{dst}/unstate.pkl")
        unstate_df = dimension_reduction(sampled_state, "TSNE", n_components=2)
        unstate_df.to_csv(fn)
        preds = model.over_predict(sampled_cases, n=over)
        thresh_items = label_output(preds)
        pred_prob = [list(zip(mlb.classes_, pred)) for pred in preds]
        eval_json = build_eval_json(sampled_cases, pred_prob, thresh_items)
        dump_json(f"{dst}/eval.json", eval_json)
    fig = state_plot(unstate_df, 12)
    fig.write_image(f"{dst}/unlabel_tsne.pdf")
    fig.write_html(f"{dst}/unlabel_tsne.html")

    # Labelled
    fn = f"{dst}/label_tsne.csv"
    if os.path.exists(fn):
        state_df = pd.read_csv(fn, index_col=0)
    else:
        states = get_tag_states(model, dataset, mlb)
        state_df = dimension_reduction(states, "TSNE", n_components=2)
        state_df.to_csv(fn)
    fig = state_plot(state_df, 12)
    fig.write_image(f"{dst}/label_tsne.pdf")
    fig.write_html(f"{dst}/label_tsne.html")

    # Performance
    eval_model(model, dataset, over, mlb, dst, "")

    # Top key
    fn = f"{dst}/top_key.json"
    if os.path.exists(fn):
        top_key = load_json(fn)
    else:
        maskExplainer = MaskExplainer(mlb)
        top = top_keywords(maskExplainer, model, dataset.x_dict)
        top_key = filter_top(top, dataset.y_tags, thresh=20)
        dump_json(fn, top_key)
    fig = kw_plot(top_key)
    fig.write_image(f"{dst}/knockout_result.pdf")


if __name__ == "__main__":
    from fire import Fire

    Fire(make_figures)
