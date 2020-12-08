import json
import pickle
import time
from pathlib import Path
from typing import List
from zipfile import ZipFile

import requests
from tqdm.autonotebook import tqdm

from .domain import DATAFILE, LabelledCase, RawData


def prepare_model(out_path="model"):
    url = "https://storage.googleapis.com/pathopatho/model.zip"
    out_f = "model.zip"
    download_file(url, out_f)
    unzip_model(out_f, out_path)


def download_file(url, out_f):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_f, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
    return out_f


def unzip_model(p, out):
    with ZipFile(p, "r") as zipfile:
        zipfile.extractall(out)
    m = Path(out)
    for f in m.glob("*/*"):
        f.rename(m / f.name)


def prepare_dataset(out_f="dataset.zip"):
    url = "https://storage.googleapis.com/pathopatho/dataset.zip"
    download_file(url, out_f)


def prepare_unlabelled(out_f="data/unlabelled.json"):
    url = "https://storage.googleapis.com/pathopatho/unlabelled.json"
    download_file(url, out_f)


def prepare_state(out_f="data/unstate.pkl"):
    url = "https://storage.googleapis.com/pathopatho/unstate.pkl"
    download_file(url, out_f)


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def load_json(path):
    with open(path, "r") as js_:
        return json.load(js_)


def dump_json(path, obj):
    with open(path, "w") as js_:
        json.dump(obj, js_, indent=4)


def load_datazip(datazip_path: str, datafile: dict = DATAFILE):
    with ZipFile(datazip_path, "r") as datazip:
        tmp = []
        for f_name in datafile.values():
            with datazip.open(f_name, "r") as file:
                tmp.append(json.loads(file.read().decode("utf-8")))
        return RawData(*tmp)


def dump_datazip(rawdata: RawData, zip_name=None):
    if zip_name is None:
        zip_name = f"dataset{get_timestamp()}.zip"
    with ZipFile(zip_name, "w") as datazip:
        for name, data in rawdata:

            with datazip.open(f"{name}.json", "w") as file:
                file.write(json.dumps(data).encode("utf-8"))
    return zip_name


def dump_labelled_cases(labelled_cases: List[LabelledCase], path: str):
    obj = list(map(LabelledCase.serialize, labelled_cases))
    dump_json(path, obj)


def load_state(state_p: str):
    with open(state_p, "rb") as plk:
        state = pickle.load(plk)
    return state


def dump_state(states, state_p="state"):
    with open(state_p, "wb") as plk:
        pickle.dump(states, plk)
    return state_p


def build_eval_json(sampled_cases, pred_prob, pred_out):
    eval_json = []
    for idx, (case, prob, pred) in enumerate(zip(sampled_cases, pred_prob, pred_out)):
        eval_json.append(
            {
                "id": idx,
                "text": case,
                "prob": [(n, str(p)) for n, p in prob],
                "pred": [bool(v) for v in pred],
            }
        )
    return eval_json
