import pickle

from tagc.io_utils import load_json
from tagc.model import StandaloneModel


def embed(model_p, case_p):
    cases = load_json(case_p)
    model = StandaloneModel.from_path(model_p)
    embed = model.predict(cases, pooled_output=True)
    with open("embed.pkl", "wb") as target:
        pickle.dump(embed, target)


if __name__ == "__main__":
    from fire import Fire

    Fire(embed)
