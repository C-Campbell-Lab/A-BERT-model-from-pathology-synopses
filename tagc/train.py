from typing import Optional

from sklearn import metrics
from transformers import BertConfig, Trainer, TrainingArguments

from .dataset import DatasetFactory
from .domain import Params
from .model import Classification
from .validation import summary


class Pipeline:
    def __init__(self, params: Params):
        self.init_model(params)
        self.init_dataset(params)

    def init_model(self, params):
        config = BertConfig()
        config.dropout_prob = params.dropout_prob
        config.num_labels = params.num_labels
        config.identifier = params.identifier
        self.config = config
        self.model = Classification(config)

    def init_dataset(self, params):
        df = DatasetFactory(params)
        self.dataset_factory = df
        self.training_set, self.testing_set = df.supply_training_dataset()
        self.mlb = df.mlb
        self.tokenizer = df.tokenizer

    def train(self, training_args: Optional[TrainingArguments] = None):
        if training_args is None:
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=10,
                per_device_train_batch_size=8,
                save_steps=1000,
                save_total_limit=2,
                evaluate_during_training=True,
                logging_dir="./logs",
                eval_steps=225,
            )

        self.model.train()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.training_set,
            eval_dataset=self.testing_set,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()
        trainer.evaluate()
        self.trainer = trainer

    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions >= 0
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

    def validation_examples(self):
        assert self.trainer is not None, "training first"
        pred = self.trainer.predict(self.testing_set)
        cases = self.dataset_factory.x_test_dict
        true_tags = self.dataset_factory.y_test_tags
        pred_tags = self.dataset_factory.mlb.inverse_transform(pred.predictions >= 0)
        example, judges_count = summary(cases, true_tags, pred_tags)
        print(judges_count)
        return example


if __name__ == "__main__":
    import fire

    print(fire.Fire(Params))
