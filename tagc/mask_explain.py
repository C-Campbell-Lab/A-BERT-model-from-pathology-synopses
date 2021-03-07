import re
from collections import defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.autonotebook import tqdm

from .data_utils import count_tags
from .domain import Case, Mask, MaskedParent, MaskRet, Trace, get_thresh
from .model import StandaloneModel, label_output


class MaskMaker:
    def __init__(self, pattern=r"\w+"):
        self.finder = re.compile(pattern)

    def __call__(self, case: Case):
        masks = []
        for field, sent in case.items():
            for match in self.finder.finditer(sent):
                start, end = match.span()
                mask = Mask(field, start, end, match.group(0))
                masks.append(mask)
        masked_parent = MaskedParent(masks, case)
        return masked_parent


class MaskExplainer:
    def __init__(
        self, mlb: MultiLabelBinarizer, mask_maker: MaskMaker = None, thresh=None
    ):
        if mask_maker is None:
            self.mask_maker = MaskMaker()
        else:
            self.mask_maker = mask_maker
        self.mlb = mlb
        self.thresh = get_thresh() if thresh is None else thresh

    def explain(self, model: StandaloneModel, case: Case):
        origin_output = model.predict([case], tqdm_disable=True)
        masked_parent = self.mask_maker(case)
        masked_cases = masked_parent.masked_cases()
        masked_outputs = model.predict(masked_cases, tqdm_disable=True)
        masks = np.array(masked_parent.masks)
        trace = Trace(origin_output, masked_outputs, masks)
        ret = self.analysis_trace(trace)
        return ret

    def analysis_trace(self, trace: Trace) -> List[MaskRet]:
        def sort_col_descend(values, col):
            return np.argsort(values[:, col])[::-1]

        pred = label_output(trace.origin_output, self.thresh)
        bool_idx = pred[0]
        trace.important_change = (
            trace.origin_output[:, bool_idx] - trace.masked_outputs[:, bool_idx]
        )
        pred_tag = self.mlb.inverse_transform(pred)[0]
        rets = []
        for idx, tag in enumerate(pred_tag):
            rank = sort_col_descend(trace.important_change, idx)
            importance = [
                (mask, value)
                for mask, value in zip(
                    trace.masks[rank], trace.important_change[:, idx][rank]
                )
            ]
            rets.append(MaskRet(tag, importance))
        self.trace = trace
        return rets

    def show_trace(self):
        trace = self.trace
        return trace.origin_output, trace.masked_outputs, trace.important_change


def plot_explanation(rets: List[MaskRet], dash=False):
    fig = make_subplots(
        rows=len(rets), cols=1, subplot_titles=tuple(ret.tag for ret in rets)
    )
    for loc, mask_ret in enumerate(rets, start=1):
        importance = mask_ret.importance
        words_p = [p[0].word for p in importance if p[1] >= 0]
        values_p = [p[1] for p in importance if p[1] >= 0]
        words_n = [p[0].word for p in importance if p[1] < 0]
        values_n = [p[1] for p in importance if p[1] < 0]

        fig.add_trace(go.Bar(x=words_p, y=values_p, name="pos"), row=loc, col=1)
        fig.add_trace(go.Bar(x=words_n, y=values_n, name="neg"), row=loc, col=1)
        # fig.update_xaxes(title_text="word", row=loc, col=1)
        fig.update_yaxes(title_text="influence", row=loc, col=1)
    fig.layout.update(showlegend=False)
    if dash:
        return fig
    fig.show()


def top_keywords(
    mask_explainer: MaskExplainer, model: StandaloneModel, cases, top_n=-1
):
    rets = collect_rets(mask_explainer, model, cases)
    keywords = sum_keywords(rets, top_n)
    return refine_top(keywords, top_n)


def collect_rets(mask_explainer: MaskExplainer, model: StandaloneModel, cases):
    rets = []
    for case in tqdm(cases):
        rets.extend(mask_explainer.explain(model, case))
    return rets


def sum_keywords(rets, top_n=5):
    dashboard: Dict[str, DefaultDict[str, float]] = {}

    for ret in rets:
        importance = ret.importance[:top_n]
        if ret.tag in dashboard:
            tmp_dict = dashboard[ret.tag]
        else:
            tmp_dict = defaultdict(float)
            dashboard[ret.tag] = tmp_dict
        for k, v in importance:
            tmp_dict[k.word] += v

    return dashboard


def refine_top(top, top_n=5):
    refine_ret = {}
    for k, v in top.items():
        crop_len = len(v) if top_n == -1 else top_n
        tmp = sorted(v.items(), key=lambda x: x[1], reverse=True)[:crop_len]
        sum_ = sum(v for _, v in tmp)
        refine_ret[k] = [(k, v / sum_) for k, v in tmp]
    return refine_ret


def filter_top(top: dict, tags, thresh=20):
    tag_counter = count_tags(tags)
    large_enough = [k for k, v in tag_counter.items() if v >= thresh]
    left = {}
    for tag, kw in top.items():
        if tag in large_enough:
            left[tag] = kw
    return left
