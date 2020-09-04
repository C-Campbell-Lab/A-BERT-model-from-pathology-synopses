from dataclasses import dataclass
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
    x_train: str
    y_train: str
    x_test: str
    y_test: str
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
