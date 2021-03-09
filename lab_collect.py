from pathlib import Path
from collections import defaultdict
from tagc.evaluation import expert_eval, form_pred
from zipfile import ZipFile
from tagc.io_utils import load_json, dump_json


def collect_ac():
    collector = defaultdict(list)
    for name in ("labF", "labS", "labT", "lab0"):
        for fix in ("", "R"):
            lab = Path(f"newLab/{name}{fix}/keepKey_200")
            for target in lab.glob("*overall.json"):
                ret = load_json(target)
                collector[str(target)].append(ret)
    dump_json("collect_ac.json", collector)


def collect_feedback():
    collector = defaultdict(list)
    for name in ("labF", "labS", "labT", "lab0"):
        for fix in ("C", "M"):
            lab = Path(f"newLab/{name}/feedback{fix}")
            for target in lab.glob("*over*.json"):
                ret = load_json(target)
                collector[str(target)].append(ret)
    dump_json("collect_feedback.json", collector)


def collect_up():
    collector = defaultdict(list)
    for name in ("labF", "labS", "labT", "lab0"):
        for fix in ("", "noUp"):
            lab = Path(f"newLab/{name}{fix}")
            for target in lab.glob("*/400*.json"):
                ret = load_json(target)
                collector[str(target)].append(ret)
    dump_json("collect_up.json", collector)


def collect_perf():
    collector = []
    for name in ("labF", "labS", "labT", "lab0"):
        lab = Path(f"newLab/{name}")
        for target in lab.glob("*/400*.csv"):
            collector.append(str(target))
    zip_name = "collect_perf.zip"
    with ZipFile(zip_name, "w") as zipFile:
        _copy_csv(zipFile, collector)


def collect_expert_eval():
    collector = {}
    for lab in (f"newLab/{name}" for name in ("labF", "labS", "labT", "lab0")):
        pred_json = f"{lab}/figs/eval.json"
        for exp in ("mona", "cathy"):
            exp_eval_csv = f"{lab}/feedback{exp[0].title()}/{exp}_j1.csv"
            eval_over = expert_eval(exp_eval_csv, form_pred(pred_json))
            collector[f"{lab}{exp}eval1"] = eval_over
    dump_json("expert_eval1.json", collector)


def _copy_csv(zipFile, collector):
    for path in collector:
        with zipFile.open(path, "w") as file, open(path, "rb") as src:
            file.write(src.read())


def main():
    collect_ac()
    collect_feedback()
    collect_up()
    collect_perf()
    collect_expert_eval()


if __name__ == "__main__":
    main()
