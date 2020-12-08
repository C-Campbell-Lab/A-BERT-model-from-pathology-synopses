from tagc.model import StandaloneModel
from tagc.io_utils import build_eval_json, load_datazip, load_json
from sklearn.preprocessing import MultiLabelBinarizer


def test_prediction():
    model = StandaloneModel.from_path("TagModel", keep_key=False, max_len=100)
    cases = load_json("data/eval.json")
    ds = load_datazip("data/dataset.zip")
    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    indexes = [9, 15]
    used = [cases[idx]["text"] for idx in indexes]
    prob = model.predict_prob(used, mlb)
    tags = model.predict_tags(used, mlb)
    pred_out = mlb.transform(tags)
    build_eval_json(used, prob, pred_out)
