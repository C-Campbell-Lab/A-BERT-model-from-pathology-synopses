from dataclasses import dataclass

import fire
from sklearn import metrics
from transformers import BertConfig, Trainer, TrainingArguments

from .dataset import supply_dataset
from .model import Classification


def make_config(dropout_prob=0.3, num_labels=18):
    config = BertConfig()
    config.dropout_prob = dropout_prob
    config.num_labels = num_labels
    config.identifier = "bert-base-uncased"
    return config


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


def fine_tuning():
    config = make_config()
    model = Classification(config)
    training_set, testing_set = supply_dataset()
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=2,
        evaluate_during_training=True,
        logging_dir="./logs",
        eval_steps=225,
        weight_decay=0.0,
        learning_rate=1e-05,
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


@dataclass
class Param:
    x_train: str
    y_train: str
    x_test: str
    y_test: str
    max_len: int
    upsumpling: int


if __name__ == "__main__":
    print(fire.Fire(Param))
