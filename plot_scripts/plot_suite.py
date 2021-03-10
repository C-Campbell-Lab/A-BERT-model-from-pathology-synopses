from .plot_ac_compare import mk_ac_df, plot_ac_compare, load_json
from .plot_feedback import mk_fd_df, plot_feedback
from .plot_upsample import mk_up_df
from .plot_tag_perf import plot_tag_perf_with_std, mk_std_perf

collect_ac = "out/collect_ac.json"
collect_feedback = "out/collect_feedback.json"
collect_up = "out/collect_up.json"
collect_perf = "out/collect_perf.zip"
outdir = "out"
ac_up_df, rd_up_df = mk_ac_df(load_json(collect_ac))
ac_fig = plot_ac_compare(ac_up_df, rd_up_df)
ac_fig.write_image(f"{outdir}/acl_result.pdf")

_, start_f1, start_err = ac_up_df.iloc[-1].to_list()
fd_df = mk_fd_df(load_json(collect_feedback))
fd_fig, fd_df_sum = plot_feedback(start_f1, start_err, fd_df)
fd_fig.write_image(f"{outdir}/review_result.pdf")
fd_df_sum.to_csv(f"{outdir}/review_result.csv")

up_df = mk_up_df(load_json("collect_up.json"))
up_df.to_csv(f"{outdir}/upsample_ret.csv")

perf_df = mk_std_perf("collect_perf.zip")
perf_fig = plot_tag_perf_with_std(perf_df, start_f1, start_err)
perf_fig.write_image(f"{outdir}/dev_result.pdf")
perf_df.to_csv(f"{outdir}/dev_result.csv")
