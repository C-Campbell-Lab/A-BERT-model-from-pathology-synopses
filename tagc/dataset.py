import json
import random
from collections import defaultdict

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

random.seed(42)


class CustomDataset(Dataset):
    def __init__(self, texts, tags, tokenizer: AutoTokenizer, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tags = tags
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.tags[index], dtype=torch.float),
        }


# Makeshift upsampling to 125 cases per tag
def grouping_idx(y):
    groupby = defaultdict(list)
    for idx, tags in enumerate(y):
        for tag in tags:
            groupby[tag].append(idx)
    return groupby


def compose(case):
    tmp = [f"{k}: {v}" for k, v in case.items()]
    random.shuffle(tmp)
    return " ".join(tmp)


def upsampling(x, y, target=100):
    groupby_idx = grouping_idx(y)
    new_x = []
    new_y = []
    for group_idx in groupby_idx.values():
        upsample_idx = random.choices(group_idx, k=target)
        new_x.extend(map(lambda idx: compose(x[idx]), upsample_idx))
        new_y.extend(map(lambda idx: y[idx], upsample_idx))
    return new_x, new_y


def load_json(path):
    with open(path, "r") as js_:
        return json.load(js_)


def supply_dataset(params):
    x_test_dict = load_json(params.x_test)
    x_train_dict = load_json(params.x_train)
    y_train_tags = load_json(params.y_train)
    y_test_tags = load_json(params.y_train)
    x_train, y_train_raw = upsampling(
        x_train_dict, y_train_tags, target=params.upsampling
    )
    y = y_train_tags + y_test_tags

    mlb = MultiLabelBinarizer().fit(y)
    y_train = mlb.transform(y_train_raw)
    x_test = list(map(compose, x_test_dict))
    y_test = mlb.transform(y_test_tags)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = CustomDataset(x_train, y_train, tokenizer, params.max_len)
    testing_set = CustomDataset(x_test, y_test, tokenizer, params.max_len)
    return train_dataset, testing_set
