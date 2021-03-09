import os


def main(
    upsample=False,
    datasize=False,
    feedback=False,
    plot=False,
    dry_run=True,
    fix="random",
):
    assert (
        sum((upsample, datasize, feedback, plot)) < 2
    ), "only one experiment suite can run at once"
    if upsample:
        upsample_exp(dry_run=dry_run)
    if datasize:
        datasize_exp(fix=fix, dry_run=dry_run)
    if feedback:
        continue_exp(dry_run=dry_run)
    if plot:
        plot_exp(dry_run=dry_run)


FMT = 'sbatch --mem=16G --cpus-per-task=8 --gres=gpu:1 --wrap="python3 {}"'


def upsample_exp(dry_run=True):
    contents = (
        f"make_exp.py {lab} --dataset_path {dsp} --upsample -200 --over -1 --step 400"
        for lab, dsp in zip(
            (f"newLab/{name}noUp" for name in ("labF", "labS", "labT")),
            (f"standardDs{idx}.zip" for idx in range(3)),
        )
    )
    _exe_content(contents, dry_run)


def datasize_exp(fix="random", dry_run=True):
    contents = (
        f"make_exp.py {lab} --dataset_path {dsp}"
        for lab, dsp in zip(
            (
                f"newLab/{name}{'' if fix != 'random' else 'R'}"
                for name in ("labF", "labS", "labT")
            ),
            (f"{fix}Ds{idx}.zip" for idx in range(3)),
        )
    )
    _exe_content(contents, dry_run)


def continue_exp(dry_run=True):
    experts = (f"{exp}_j.csv" for exp in ("mona", "cathy"))
    contents = (
        (
            f"feedback.py {lab}/keepKey_200/model/ --eval_ret {expert} --dataset_p {dsp} --outdir {lab}/feedback{expert[0].title()}"
            for lab, dsp in zip(
                (f"newLab/{name}" for name in ("labF", "labS", "labT")),
                (f"standardDs{idx}.zip" for idx in range(3)),
            )
        )
        for expert in experts
    )
    for content_g in contents:
        _exe_content(content_g, dry_run)


def plot_exp(dry_run=True):
    unlabelled_p = "unlabelled.json"
    contents = (
        f"train.py {lab}/keepKey_200/model/ {dsp} {unlabelled_p} --outdir {lab}/figs/ --train False --plot"
        for lab, dsp in zip(
            (f"newLab/{name}" for name in ("labF", "labS", "labT")),
            (f"standardDs{idx}.zip" for idx in range(3)),
        )
    )
    _exe_content(contents, dry_run)


def _exe_content(content_g, dry_run):
    for content in content_g:
        cmd = FMT.format(content)
        if dry_run:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
