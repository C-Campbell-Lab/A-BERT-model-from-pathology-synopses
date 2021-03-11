from pathlib import Path

from tagc.make_ds import mk_cases, mk_randomDs, mk_standardDs, mk_unlabelled


def main(xlsx_p, final_dsp, tmp_dsp, eval_ret):
    dst = Path("data")
    dst.mkdir(exist_ok=True)
    cases_p = mk_cases(xlsx_p, case_p=str(dst / "cases.json"))
    unlabelled_p = mk_unlabelled(
        final_dsp=final_dsp, cases_p=cases_p, outpath=str(dst / "unlabelled.json")
    )
    standard_dsps = mk_standardDs(final_dsp=tmp_dsp, dst=str(dst), plot=True)
    mk_randomDs(
        standard_dsps,
        eval_ret=eval_ret,
        unlabelled_p=unlabelled_p,
        dst=str(dst),
        plot=True,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
