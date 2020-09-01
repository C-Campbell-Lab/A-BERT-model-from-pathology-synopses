from sklearn import metrics
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    Trainer,
    TrainingArguments,
)

from .dataset import supply_dataset


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


class Classification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            loss = loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions >= 0.5
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = metrics.accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def fine_tuning(hidden_dropout_prob=0.3, num_labels=18):
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        hidden_dropout_prob=hidden_dropout_prob,
        num_labels=num_labels,
    )
    model = Classification(config)

    training_set, testing_set = supply_dataset()
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=7,
        per_device_train_batch_size=16,
        save_steps=1000,
        save_total_limit=2,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=testing_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
