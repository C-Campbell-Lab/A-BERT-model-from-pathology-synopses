import pandas as pd


def mk_up_df(up_history: dict, do_print=True):
    f1_scores = []
    upsample = []
    labs = []
    for k, v in up_history.items():
        upsample.append(False if "-200" in k else True)
        f1_scores.append(v[0]["F1 Score"])
        labs.append(k.split("/")[1])
    df = pd.DataFrame(dict(f1_scores=f1_scores, upsample=upsample, labs=labs))
    if do_print:
        print(df.groupby("upsample")["f1_scores"].agg(["mean", "std"]).to_latex())
    return df
