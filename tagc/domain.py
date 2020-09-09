from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

Case = dict


@dataclass
class LabelledCase:
    text: Case
    tag: List[str]

    def serialize(self):
        return {"text": self.text, "tag": "; ".join(self.tag)}


@dataclass
class Mask:
    field: str
    start: int
    end: int

    def __call__(self, case: Case):
        case_copy = case.copy()
        field = case_copy[self.field]
        case_copy[self.field] = field[: self.start] + field[self.end :]
        return case_copy

    def word(self, case: Case):
        return case[self.field][self.start : self.end]


@dataclass
class MaskedParent:
    masks: List[Mask]
    text: Case

    def masked_cases(self):
        return [mask(self.text) for mask in self.masks]

    def mask_words(self):
        return [mask.word(self.text) for mask in self.masks]


@dataclass
class Params:
    datazip_path: str
    max_len: int
    upsampling: int
    dropout_prob: float
    num_labels: int
    identifier: str


@dataclass
class Trace:
    origin_output: np.array
    masked_outputs: np.array
    mask_words: np.array
    important_change: Optional[np.array] = None


@dataclass
class States:
    data: np.array
    tag: list
    index: list
    tag_n: list


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
    x_dict: dict
    y_tags: list
    x_train_dict: dict
    y_train_tags: list
    x_test_dict: dict
    y_test_tags: list

    def __iter__(self):
        return iter(asdict(self).items())
