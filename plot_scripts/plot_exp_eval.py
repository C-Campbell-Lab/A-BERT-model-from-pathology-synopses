import pandas as pd


def mk_eval_df(eval_history: dict, do_print=True):
    f1_scores = []
    experts = []
    labs = []
    batches = []
    for k, v in eval_history.items():
        parts = k.split("/")[-1].split("_")
        labs.append(parts[0])
        experts.append(parts[1])
        batches.append(int(parts[-1]))
        f1_scores.append(list(v.values())[0]["f1"])
    df = pd.DataFrame(
        dict(batches=batches, experts=experts, labs=labs, f1_scores=f1_scores)
    )
    if do_print:
        print(
            df.groupby("batches")["f1_scores"]
            .agg(["mean", "std"])
            .to_latex(float_format=lambda x: "%.3f" % x)
        )
    return df
