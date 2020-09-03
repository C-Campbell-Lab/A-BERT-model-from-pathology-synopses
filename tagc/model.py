from typing import Any, Dict, Optional, Union

import torch
from torch import cuda, nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


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
    def __init__(self, model, tokenizer, max_len=200):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, texts: list, batch=8):
        self.model.eval()
        preds: Optional[torch.Tensor] = None
        for step in range(0, len(texts), batch):
            inputs = self._encode(texts[step : step + batch])
            logits = self._predict_step(inputs)
            preds = logits if preds is None else torch.cat((preds, logits), dim=0)
        if preds is None:
            return preds
        return torch.sigmoid(preds).cpu().numpy()

    def _encode(self, texts):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )

    def _predict_step(self, inputs):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
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
