from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile
from tagc.io_utils import load_json, dump_json


def collect_ac():
    collector = defaultdict(list)
    for name in ("labF", "labS", "labT"):
        for fix in ("", "R"):
            lab = Path(f"newLab/{name}{fix}/keepKey_200")
            for target in lab.glob("*overall.json"):
                ret = load_json(target)
                collector[str(target)].append(ret)
    dump_json("collect_ac.json", collector)


def collect_feedback():
    collector = defaultdict(list)
    for name in ("labF", "labS", "labT"):
        for fix in ("C", "M"):
            lab = Path(f"newLab/{name}/feedback{fix}")
            for target in lab.glob("*over*.json"):
                ret = load_json(target)
                collector[str(target)].append(ret)
    dump_json("collect_feedback.json", collector)


def collect_up():
    collector = defaultdict(list)
    for name in ("labF", "labS", "labT"):
        for fix in ("", "noUp"):
            lab = Path(f"newLab/{name}{fix}")
            for target in lab.glob("*/400*.json"):
                ret = load_json(target)
                collector[str(target)].append(ret)
    dump_json("collect_up.json", collector)


def collect_perf():
    collector = []
    for name in ("labF", "labS", "labT"):
        lab = Path(f"newLab/{name}")
        for target in lab.glob("*/400*.csv"):
            collector.append(str(target))
    zip_name = "collect_perf.zip"
    with ZipFile(zip_name, "w") as zipFile:
        _copy_csv(zipFile, collector)


def _copy_csv(zipFile, collector):
    for path in collector:
        with zipFile.open(path, "w") as file, open(path, "rb") as src:
            file.write(src.read())


def main():
    collect_ac()
    collect_feedback()
    collect_up()
    collect_perf()


if __name__ == "__main__":
    main()
