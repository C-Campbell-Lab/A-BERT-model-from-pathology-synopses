from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import cuda, nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .data_utils import compose
from .domain import Mlb, get_thresh


class Classification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(config.identifier)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class StandaloneModel:
    def __init__(
        self, model: Classification, tokenizer=None, max_len=100, keep_key=False
    ):
        self.model = model
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.keep_key = keep_key
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model.to(self.device)

    @classmethod
    def from_path(cls, model_path: str, tokenizer=None, max_len=100, keep_key=False):
        model = Classification.from_pretrained(model_path)
        return cls(model, tokenizer, max_len, keep_key)

    def over_predict(self, cases: List[dict], batch_size=8, n=3, tqdm_disable=True):
        preds = None
        for _ in range(n):
            texts = [
                compose(case, keep_key=self.keep_key, shuffle=True) for case in cases
            ]
            p = self.predict(texts, batch_size=batch_size, tqdm_disable=tqdm_disable)
            preds = p if preds is None else np.maximum(preds, p)
        return preds

    def over_predict_tags(
        self, cases: list, mlb: Mlb, batch_size=8, tqdm_disable=True, thresh=None, n=3
    ) -> list:
        if n == -1:
            preds = self.predict(
                cases, batch_size=batch_size, tqdm_disable=tqdm_disable
            )
        else:
            preds = self.over_predict(
                cases, batch_size=batch_size, tqdm_disable=tqdm_disable, n=n
            )
        thresh_items = label_output(preds, thresh)
        return mlb.inverse_transform(thresh_items)

    def predict_tags(
        self, cases: list, mlb: Mlb, batch_size=8, tqdm_disable=True, thresh=None
    ) -> list:
        preds = self.predict(cases, batch_size=batch_size, tqdm_disable=tqdm_disable)
        thresh_items = label_output(preds, thresh)
        return mlb.inverse_transform(thresh_items)

    def predict_prob(
        self, cases: list, mlb: Mlb, batch_size=8, tqdm_disable=True
    ) -> List[List[Tuple[str, float]]]:
        preds = self.predict(cases, batch_size=batch_size, tqdm_disable=tqdm_disable)
        return [list(zip(mlb.classes_, pred)) for pred in preds]

    def predict(
        self, cases: list, batch_size=8, pooled_output=False, tqdm_disable=False
    ) -> np.array:
        """Predict the prob for 21 tags. If cases are dict, they will be
        composed by their `values` without shuffle.

        Args:
            cases (list): The input, can be either a list of str or a list of dict.
            batch_size (int, optional): The batch size. Defaults to 8.
            pooled_output (bool, optional): Whether set the output as 768-D vector. Defaults to False.

        Returns:
            np.array: [description]
        """
        pooled_input = False
        if isinstance(cases[0], dict):
            cases = list(map(self._compose, cases))
        elif isinstance(cases[0], np.ndarray):
            pooled_input = True
        self.model.eval()
        preds: Optional[torch.Tensor] = None
        case_size = len(cases)
        for step in tqdm(range(0, case_size, batch_size), disable=tqdm_disable):
            batch = cases[step : step + batch_size]
            if pooled_input:
                outputs = self._predict_on_pooled_output(
                    torch.from_numpy(batch).to(self.device, dtype=torch.float)
                )
            else:
                inputs = self._encode(batch)
                outputs = self._predict_step(inputs, pooled_output=pooled_output)
            preds = outputs if preds is None else torch.cat((preds, outputs), dim=0)
        if preds is None:
            return preds
        if pooled_output:
            return preds.cpu().numpy()
        return torch.sigmoid(preds).cpu().numpy()

    def _encode(self, texts):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )

    def _predict_step(self, inputs, pooled_output=False):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if pooled_output:
                outputs = self.model.base_model(**inputs)
                pooled_output = outputs[1]
                return pooled_output.detach()
            outputs = self.model(**inputs)
            logits = outputs[0]
        return logits.detach()

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long).to(
                self.device, dtype=torch.long
            )

        return inputs

    def _predict_on_pooled_output(self, pooled_output: np.array):
        with torch.no_grad():
            logits = self.model.classifier(pooled_output)
        return logits.detach()

    def _compose(self, case):
        return compose(case, keep_key=self.keep_key, shuffle=False)


def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def label_output(preds: np.array, thresh=None):
    if thresh is None:
        thresh = get_thresh()
    thresh_items = preds >= thresh
    for idx, thresh_item in enumerate(thresh_items):
        if sum(thresh_item) == 0:
            ix = np.argmax(preds[idx])
            thresh_items[idx][ix] = True

    return thresh_items
