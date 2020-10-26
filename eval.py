from tagc.model import StandaloneModel, label_output
from tagc.io_utils import (
    build_eval_json,
    dump_json,
    dump_state,
    load_datazip,
    load_json,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tagc.validation import dimension_reduction, get_unlabelled_state

model = StandaloneModel.from_path("TagModel", keep_key=False, max_len=100)
unlabelled_p = "outputs/unlabelled.json"
sampled_cases = load_json(unlabelled_p)
ds = load_datazip("dataset.zip")
mlb = MultiLabelBinarizer().fit(ds.y_tags)
sampled_state = get_unlabelled_state(model, sampled_cases, mlb)
dump_state(sampled_state, state_p="outputs/unstate.pkl")
unstate_df = dimension_reduction(sampled_state, "TSNE", n_components=2)
unstate_df.to_csv("outputs/unlabel_tsne.csv")
preds = model.over_predict(sampled_cases, n=5)
thresh_items = label_output(preds)
pred_prob = [list(zip(mlb.classes_, pred)) for pred in preds]
eval_json = build_eval_json(sampled_cases, pred_prob, thresh_items)
dump_json("outputs/eval.json", eval_json)
