from tagc.io_utils import load_datazip, load_json
from tagc.model import StandaloneModel
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from pathlib import Path


def prediction(case_p: str, model_p: str, ds_p: str):
    """case has a id field and a text field"""
    cases = load_json(case_p)
    model = StandaloneModel.from_path(model_p, keep_key=True, max_len=150)
    ds = load_datazip(ds_p)
    mlb = MultiLabelBinarizer().fit(ds.y_tags)
    over = 5
    texts = cases["text"]
    ids = cases["id"]
    preds = model.over_predict_tags(texts, mlb, n=over)
    df = pd.DataFrame({"tags": preds, "id": ids})
    df.to_csv(Path(case_p).stem + "pred_tags.csv")


if __name__ == "__main__":
    from fire import Fire

    Fire(prediction)
