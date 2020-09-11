import re

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from .domain import Case, Mask, MaskedParent, Trace
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
        masked_parent = MaskedParent(masks, case)
        return masked_parent


class MaskExplainer:
    def __init__(self, mlb: MultiLabelBinarizer, mask_maker: MaskMaker = None):
        if mask_maker is None:
            self.mask_maker = MaskMaker()
        else:
            self.mask_maker = mask_maker
        self.mlb = mlb

    def explain(self, model: StandaloneModel, case: Case):
        origin_output = model.predict([case])
        masked_parent = self.mask_maker(case)
        masked_cases = masked_parent.masked_cases()
        masked_outputs = model.predict(masked_cases)

        mask_words = np.array(masked_parent.mask_words())

        trace = Trace(origin_output, masked_outputs, mask_words)
        ret = self.analysis_trace(trace)
        return ret

    def analysis_trace(self, trace: Trace):
        pred = trace.origin_output >= 0.5
        bool_idx = pred[0]
        trace.important_change = (
            trace.origin_output[:, bool_idx] - trace.masked_outputs[:, bool_idx]
        )
        pred_tag = self.mlb.inverse_transform(pred)[0]
        ret = {}
        for idx, tag in enumerate(pred_tag):
            rank = np.argsort(trace.important_change[:, idx])[::-1]
            pairs = [
                (word, value)
                for word, value in zip(
                    trace.mask_words[rank], trace.important_change[:, idx][rank]
                )
            ]
            ret[tag] = pairs
        self.trace = trace
        return ret

    def show_trace(self):
        trace = self.trace
        return trace.origin_output, trace.masked_outputs, trace.important_change


# TODO
# Top 5 keywords for each tags
