import re

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from .domain import Case, Mask, MaskedCase
from .model import StandaloneModel


class MaskMaker:
    def __init__(self, pattern=r"\w+"):
        self.finder = re.compile(pattern)

    def __call__(self, case: Case):
        masks = []
        for field, sent in case.items():
            for match in self.finder.finditer(sent):
                start, end = match.span()
                mask = Mask(field, start, end)
                masks.append(mask)
        masked_case = MaskedCase(masks, case)
        return masked_case


class MaskExplainer:
    def __init__(self, mask_maker: MaskMaker, mlb: MultiLabelBinarizer):
        self.mask_maker = mask_maker
        self.mlb = mlb

    def explain(self, model: StandaloneModel, case: Case):
        primary = model.predict([self._compose(case)])
        masked_case = self.mask_maker(case)
        cases = masked_case.masked_cases()
        texts = list(map(self._compose, cases))
        outputs = model.predict(texts)

        pred = primary >= 0.5
        bool_idx = pred[0]
        important_change = primary[:, bool_idx] - outputs[:, bool_idx]
        mask_words = np.array(masked_case.mask_words())

        pred_tag = self.mlb.inverse_transform(pred)[0]
        ret = {}
        for idx, tag in enumerate(pred_tag):
            rank = np.argsort(important_change[:, idx])[::-1]
            pairs = [
                (word, value)
                for word, value in zip(mask_words[rank], important_change[:, idx][rank])
            ]
            ret[tag] = pairs
        return ret

    def _compose(self, case: dict):
        return " ".join(f"{k}: {v}" for k, v in case.items())
