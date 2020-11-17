from pathlib import Path
from collections import defaultdict
from tagc.io_utils import load_json, dump_json
from data_size import run_exp


def main(run_metrics=True):
    for over in (-1, 5):
        paths = [Path(f"lab{i}") for i in range(1, 4)]
        collector = defaultdict(list)
        for p in paths:
            if run_metrics:
                run_exp(p, metrics_only=True, over=over)
            for target in p.glob(f"*/*{over}*.json"):
                ret = load_json(target)
                collector[target.parent.name].append(ret)

        dump_json(f"lab_ret{over}.json", collector)


if __name__ == "__main__":
    main()
