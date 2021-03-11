from tagc.data_pipe import (
    dataset_split,
    form_random_ds,
    sample_evaluation_from_path,
    xlsx_to_cases,
)
from tagc.data_utils import rawdata_stat
from tagc.io_utils import dump_json, load_datazip
from tagc.visualization import plot_tag_stat

CASE_NUM = 11418


def mk_cases(xlsx_p, case_p="cases.json"):
    cases = xlsx_to_cases(xlsx_p)
    assert len(cases) == CASE_NUM
    dump_json(case_p, cases)
    return case_p


def mk_unlabelled(
    final_dsp="standardDs.zip", cases_p="cases.json", outpath="unlabelled.json"
):
    sampled_cases = sample_evaluation_from_path(cases_p, final_dsp)
    dump_json(outpath, sampled_cases)
    return outpath


def mk_standardDs(final_dsp="standardDsTmp.zip", dst="out/standard", plot=True):
    dsps = dataset_split(final_dsp, dst)
    if plot:
        for idx, dsp in enumerate(dsps):
            rawdata = load_datazip(dsp)
            tag_stat = rawdata_stat(rawdata)
            tag_stat.to_csv(f"{dst}/data_stat{idx}.csv")
            fig = plot_tag_stat(tag_stat)
            fig.write_image(f"{dst}/data_stat{idx}.pdf")
    return dsps


def mk_randomDs(
    standard_dsps,
    eval_ret="data/evaluation/mona_j.csv",
    unlabelled_p="data/unlabelled.json",
    dst="out/random",
    plot=True,
):
    dsps = form_random_ds(standard_dsps, eval_ret, unlabelled_p, outdir=dst)
    if plot:
        for idx, dsp in enumerate(dsps):
            rawdata = load_datazip(dsp)
            tag_stat = rawdata_stat(rawdata)
            tag_stat.to_csv(f"{dst}/random_data_stat{idx}.csv")
            fig = plot_tag_stat(tag_stat)
            fig.write_image(f"{dst}/random_data_stat{idx}.pdf")
    return dsps
