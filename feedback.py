import gc
import shutil
from pathlib import Path

import pandas as pd
import torch
from make_evaluation import form_eval
from sklearn.preprocessing import MultiLabelBinarizer

from tagc.domain import Params
from tagc.evaluation import continue_eval, form_pred
from tagc.io_utils import dump_datazip, dump_json, load_datazip, load_json
from tagc.model import Classification, StandaloneModel
from tagc.train import Pipeline
from tagc.validation import eval_model


class ContinueTrainer:
    def __init__(
        self,
        init_model_p,
        eval_ret="mona_j.csv",
        dataset_p="stdDs.zip",
        ori_eval_p="outputsS/eval.json",
        unlabelled_p="outputsK/unlabelled.json",
        outdir="feedbackM",
    ):
        self.init_model_p = init_model_p
        self.eval_ret = eval_ret
        self.dataset_p = dataset_p
        self.ori_eval_p = ori_eval_p
        self.unlabelled_p = unlabelled_p
        self.outdir = outdir

        Path(self.outdir).mkdir(exist_ok=True)
        shutil.copyfile(self.dataset_p, f"{outdir}/dataset0.zip")
        shutil.copyfile(self.ori_eval_p, f"{outdir}/eval0.json")
        self.ds = load_datazip(self.dataset_p)
        self.mlb = MultiLabelBinarizer().fit(self.ds.y_tags)
        self.df = pd.read_csv(eval_ret).drop_duplicates(
            subset=["ID", "Judge"], keep="last"
        )

    def run(self, batch_size):
        outdir = self.outdir
        eval_stem = Path(self.eval_ret).stem
        for idx, step in enumerate(range(0, len(self.df), batch_size), start=1):
            batch_df = self.df.iloc[step : step + batch_size]
            if len(batch_df) < batch_size:
                break
            batch_eval_p = f"{outdir}/{eval_stem}{idx}.csv"
            batch_df.to_csv(batch_eval_p, index=None)
            base_path = f"{outdir}/dataset{idx-1}.zip"
            eval_json = f"{outdir}/eval{idx-1}.json"
            self._step(batch_eval_p, base_path, eval_json, idx)

    def _step(self, batch_eval_p, base_path, eval_json, idx_marker):
        outdir = self.outdir
        eval_over = continue_eval(batch_eval_p, form_pred(eval_json))
        dump_json(f"{outdir}/eval_over{idx_marker}.json", eval_over)
        ds = self._add_training(
            batch_eval_p,
            base_path,
            idx_marker=idx_marker,
        )
        self._continue_train(ds, idx_marker=idx_marker)
        eval_over = continue_eval(
            batch_eval_p, form_pred(f"{outdir}/eval{idx_marker}.json")
        )
        dump_json(f"{outdir}/eval_over_after{idx_marker}.json", eval_over)

    def _add_training(
        self,
        eval_ret: str,
        base_path,
        idx_marker=1,
    ):
        outdir = self.outdir
        unlabelled_p = self.unlabelled_p
        ds = load_datazip(base_path)
        df = pd.read_csv(eval_ret).drop_duplicates(subset=["ID", "Judge"], keep="last")
        indices = df["ID"].to_list()
        sampled_cases = load_json(unlabelled_p)
        add_texts = [sampled_cases[idx] for idx in indices]
        y_true_ = df["eval"].map(lambda x: x.split(", ")).to_list()

        ds.x_train_dict = ds.x_train_dict + add_texts
        ds.y_train_tags = ds.y_train_tags + y_true_
        ds.x_dict = ds.x_dict + add_texts
        ds.y_tags = ds.y_tags + y_true_
        dsp = dump_datazip(ds, f"{outdir}/dataset{idx_marker}.zip")
        print(dsp)
        return ds

    def _continue_train(
        self,
        rawdata,
        over=5,
        epoch=10,
        idx_marker=1,
    ):
        outdir = self.outdir
        params = Params(
            rawdata, 150, 200, 0.5, "bert-base-uncased", True, epoch, self.mlb
        )
        pipeline = Pipeline(params)
        pipeline.model = Classification.from_pretrained(self.init_model_p)
        model_p = f"{outdir}/model"
        pipeline.train(output_dir=model_p)
        pipeline.trainer.save_model(f"{outdir}/model{idx_marker}")
        standalone_model = StandaloneModel(pipeline.model, max_len=150, keep_key=True)
        form_eval(
            standalone_model,
            pipeline.mlb,
            outdir=outdir,
            marker=str(idx_marker),
            skip_state=True,
        )
        eval_model(
            standalone_model,
            rawdata,
            over,
            pipeline.mlb,
            outdir,
            len(rawdata.x_train_dict),
        )

        del pipeline
        del standalone_model
        shutil.rmtree(model_p)
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


def main(
    init_model_p,
    eval_ret="mona_j.csv",
    dataset_p="stdDs.zip",
    ori_eval_p="outputsS/eval.json",
    unlabelled_p="outputsK/unlabelled.json",
    outdir="feedbackM",
    batch_size=200,
):
    trainer = ContinueTrainer(
        init_model_p,
        eval_ret=eval_ret,
        dataset_p=dataset_p,
        ori_eval_p=ori_eval_p,
        unlabelled_p=unlabelled_p,
        outdir=outdir,
    )
    trainer.run(batch_size)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
