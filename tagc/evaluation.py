import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import sys


def main():
    df = pd.read_csv(sys.argv[1]).drop_duplicates(subset=["ID", "Judge"], keep="last")

    mlb = MultiLabelBinarizer()
    y_true_ = df["eval"].map(lambda x: x.split(", ")).to_list()
    y_pred_ = df["pred"].map(lambda x: x.split(", ")).to_list()
    y_true = mlb.fit_transform(y_true_)
    y_pred = mlb.transform(y_pred_)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    print({"precision": precision, "recall": recall, "f1": f1})


if __name__ == "__main__":
    main()
