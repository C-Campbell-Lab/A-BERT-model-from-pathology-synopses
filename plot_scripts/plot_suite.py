from plot_ac_compare import mk_ac_df, plot_ac_compare, load_json
from plot_feedback import mk_fd_df, plot_feedback
from plot_upsample import mk_up_df
from plot_tag_perf import plot_tag_perf_with_std, mk_std_perf, tag_perf_latex

collect_ac = "out/collect/collect_ac.json"
collect_feedback = "out/collect/collect_feedback.json"
collect_up = "out/collect/collect_up.json"
collect_perf = "out/collect/collect_perf.zip"
outdir = "out"
ac_up_df, rd_up_df, ac_df = mk_ac_df(load_json(collect_ac), return_df=True)
ac_df.to_csv(f"{outdir}/active_learning_result.csv")
ac_fig = plot_ac_compare(ac_up_df, rd_up_df)
ac_fig.write_image(f"{outdir}/figure2b.pdf")  # acl_result

_, start_f1, start_err = ac_up_df.iloc[-1].to_list()
fd_df = mk_fd_df(load_json(collect_feedback))
fd_fig, fd_df_sum = plot_feedback(start_f1, start_err, fd_df)
fd_fig.write_image(f"{outdir}/figure4b.pdf")  # review_result
fd_df.to_csv(f"{outdir}/review_result.csv")
fd_df_sum.to_csv(f"{outdir}/review_result_agg.csv")

up_df = mk_up_df(load_json(collect_up))
up_df.to_csv(f"{outdir}/upsample_result.csv")

perf_df, perf_df_agg = mk_std_perf(collect_perf)
perf_fig = plot_tag_perf_with_std(perf_df_agg, start_f1, start_err)
tag_perf_latex(perf_df_agg)
perf_fig.write_image(f"{outdir}/figure4a.pdf")
perf_df.to_csv(f"{outdir}/dev_result.csv")
perf_df_agg.to_csv(f"{outdir}/dev_result_agg.csv")
