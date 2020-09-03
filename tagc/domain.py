from dataclasses import dataclass
from typing import List

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

    def masking(self, case: Case):
        field = case[self.field]
        case[self.field] = field[: self.start] + field[self.end :]
        return case

    def word(self, case: Case):
        return case[self.field][self.start : self.end]


@dataclass
class MaskedCase:
    mask: Mask
    text: Case

    def masked_text(self):
        return self.mask(self.text)


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
