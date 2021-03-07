import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

Case = dict
Cases = List[dict]
Mlb = MultiLabelBinarizer
Tag = str
Label = List[Tag]
Labels = List[Label]

TAG = [
    "acute leukemia",
    "acute lymphoblastic leukemia",
    "acute myeloid leukemia",
    "acute promyelocytic leukemia",
    "basophilia",
    "chronic myeloid leukemia",
    "eosinophilia",
    "erythroid hyperplasia",
    "granulocytic hyperplasia",
    "hemophagocytosis",
    "hypercellular",
    "hypocellular",
    "inadequate",
    "iron deficiency",
    "lymphoproliferative disorder",
    "mastocytosis",
    "metastatic",
    "myelodysplastic syndrome",
    "myeloproliferative neoplasm",
    "normal",
    "plasma cell neoplasm",
]


def get_thresh():
    try:
        with open("thresh.json") as target:
            print("Use thresh.json")
            THRESH = json.load(target)
    except FileNotFoundError:
        print("Use 0.5 Threshold")
    THRESH = 0.5
    return THRESH


@dataclass
class LabelledCase:
    text: Case
    tag: Label

    def serialize(self):
        return {"text": self.text, "tag": "; ".join(self.tag)}


@dataclass
class Mask:
    field: str
    start: int
    end: int
    word: str

    def __call__(self, case: Case):
        case_copy = case.copy()
        field = case_copy[self.field]
        case_copy[self.field] = field[: self.start] + field[self.end :]
        return case_copy

    def mark(self, case: Case, mark):
        case_copy = case.copy()
        field = case_copy[self.field]
        case_copy[self.field] = field[: self.start] + mark + field[self.start :]
        return case_copy


@dataclass
class MaskedParent:
    masks: List[Mask]
    text: Case

    def masked_cases(self):
        return [mask(self.text) for mask in self.masks]

    def mask_words(self) -> List[str]:
        return [mask.word for mask in self.masks]

    def mask_words_field(self) -> List[tuple]:
        return [(mask.word, mask.field) for mask in self.masks]


@dataclass
class MaskRet:
    tag: str
    importance: list


@dataclass
class Trace:
    origin_output: np.array
    masked_outputs: np.array
    masks: np.array
    important_change: Optional[np.array] = None


@dataclass
class States:
    data: np.array
    tag: list
    index: list
    tag_n: list
    from_: list
    pred_tag: list


DATAFILE = {
    "x_dict": "x_dict.json",
    "y_tags": "y_tags.json",
    "x_train_dict": "x_train_dict.json",
    "y_train_tags": "y_train_tags.json",
    "x_test_dict": "x_test_dict.json",
    "y_test_tags": "y_test_tags.json",
}


@dataclass
class RawData:
    x_dict: Cases
    y_tags: Labels
    x_train_dict: Cases
    y_train_tags: Labels
    x_test_dict: Cases
    y_test_tags: Labels

    def __iter__(self):
        return iter(asdict(self).items())

    def show(self, from_: str, idx: int):

        print(json.dumps(self.retrive(from_, idx), indent=2))

    def retrive(self, from_: str, idx: int):
        if from_ == "train":
            x = self.x_train_dict[idx]
            y = self.y_train_tags[idx]
        else:
            x = self.x_test_dict[idx]
            y = self.y_test_tags[idx]
        return {"text": x, "tag": y}

    def to_labelled_cases(self):
        return [LabelledCase(text, tag) for text, tag in zip(self.x_dict, self.y_tags)]


@dataclass
class Params:
    raw_data: RawData
    max_len: int
    upsampling: int
    dropout_prob: float
    identifier: str
    keep_key: bool
    epoch: int
    mlb: Optional[MultiLabelBinarizer] = None
