import pickle

from tagc.io_utils import load_json
from tagc.model import StandaloneModel

BEST_MODEL_P = "labF/keepKey_200/model/"


def embed(case_p):
    cases = load_json(case_p)
    model = StandaloneModel.from_path(BEST_MODEL_P)
    embed = model.predict(cases, pooled_output=True)
    with open("embed.pkl", "wb") as target:
        pickle.dump(embed, target)


if __name__ == "__main__":
    embed("cases.json")
