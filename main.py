import gc
from copy import copy

import torch
from sklearn.model_selection import KFold

from tagc.domain import Params
from tagc.io_utils import dump_json, load_datazip
from tagc.model import StandaloneModel
from tagc.train import Pipeline


def main(dataset_path):
    ds = load_datazip(dataset_path)
    kf = KFold(n_splits=5)
    cv_result = []
    for train, test in kf.split(ds.x_train_dict):
        tmp_ds = copy(ds)
        tmp_ds.x_train_dict = [ds.x_train_dict[idx] for idx in train]
        tmp_ds.y_train_tags = [ds.y_train_tags[idx] for idx in train]
        tmp_ds.x_test_dict = [ds.x_train_dict[idx] for idx in test]
        tmp_ds.y_test_tags = [ds.y_train_tags[idx] for idx in test]
        params = Params(tmp_ds, 100, 200, 0.5, "bert-base-uncased", False, 6)
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


if __name__ == "__main__":
    import fire

    fire.Fire(main)
