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
