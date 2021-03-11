# Generate semantic embeddings from pathology synopses

This a Github Repo hosting custom codes for the paper "*A BERT model generates diagnostically relevant semantic embeddings from pathology synopses with active learning*".

## System requirements

### Hardware Requirements

For optimal performance, we recommend a computer with the following specs:

* RAM: 16+ GB
* CPU: 2+ cores, 2.2+ GHz/core
* GPU: 16+ GB

The runtimes below are generated using a computer with the recommended specs:
 * RAM: 16 GB
 * CPU: 2 Intel(R) Xeon(R) CPU @ 2.20GHz
 * GPU: 1 Tesla V100-SXM2-16GB, CUDA Version: 10.1

### Software Requirements

The package development version is tested on Linux operating system (Ubuntu 18.04.5 LTS).

Python Dependencies:

    python = "^3.6.9"
    pandas = "^1.1.3"
    transformers = "3.5.1"
    torch = "^1.7.0"
    scikit-learn = "^0.23.2"
    tqdm = "^4.50.1"
    dash = "^1.16.2"
    requests = "^2.24.0"
    plotly = "^4.11.0"
    wheel = "^0.35.1"
    fire = "^0.3.1"
    toolz = "^0.11.1"
    openpyxl = "^3.0.6"
    kaleido = "^0.2.1"

## Installation guide

    !pip -q install tagc --upgrade

*It takes about 2-5 mins.*

## Demo

> The data that support the findings of this study are available on reasonable request from the corresponding author, pending local REB and privacy office approval. The data are not publicly available because they contain information that could compromise research participant privacy/consent.

**You need first to contact the corresponding author to get the dataset "standardDs.zip" and "unlabelled.json"**

### Colab

**https://colab.research.google.com/drive/1wLzsaWWgPkoRgrUsL94Iv8_FiSiA7sKF?usp=sharing**


### Results

You can check the results in the Colab notebook above. The running time of training a model is about 10-15 mins.

## Instructions for use

### Colab
We recommend you follow the instructions in the Colab notebook above.

### Scripts

Clone this Repo and make it as the working folder (CD).

#### Dataset Creation by MCCV

    python3 make_dataset.py [xlsx_path] [final_dataset_path][tmp_dataset_path] [review_result]

For example:

    python3 make_dataset.py report.xlsx standardDs.zip standardDsTmp.zip mona_j.csv

Then, the outputs are in the **data** folder.
#### Train a model

    python3 train.py [dataset_path] [unlabelled_path] [model_path] [output_path] [--plot True] [--train True]

For example:

    python3 train.py standardDs.zip unlabelled.json out/model out --plot True --train True

Then, the model is in the **out/model** folder and its figuers are in **out** folder
#### Active learning comparison

Models trained on data sampled by active learning.

    python3 make_exp.py lab0 --dataset_path standardDs.zip

To run the experiments 3 times more, you need to change the standardDs.zip to standardDs0.zip, standardDs1.zip or standardDs2.zip and run:

    python3 make_exp.py labF --dataset_path standardDs0.zip
    python3 make_exp.py labS --dataset_path standardDs1.zip
    python3 make_exp.py labT --dataset_path standardDs2.zip

Models trained on data sampled by random selection

    python3 make_exp.py lab0R --dataset_path randomDs.zip

To run the experiments 3 times more, you need to change the randomDs.zip to randomDs0.zip, randomDs1.zip or randomDs2.zip and run:

    python3 make_exp.py labFR --dataset_path randomDs0.zip
    python3 make_exp.py labSR --dataset_path randomDs1.zip
    python3 make_exp.py labTR --dataset_path randomDs2.zip

The results are in the _lab*_ folders.

#### Improvement from feedback

    python3 feedback.py [model_path] --eval_ret [review_result] --dataset_p [dataset_path] --ori_eval_p [eval_json_path] --unlabelled_p [unlabelled_json_path]

For example,

    python3 feedback.py newLab/lab0/keepKey_200/model/ --eval_ret mona_j.csv --dataset_p standardDs.zip --ori_eval_p newLab/lab0/figs/eval.json --outdir newLab/lab0/feedbackM --unlabelled_p unlabelled.json

    python3 feedback.py newLab/lab0/keepKey_200/model/ --eval_ret cathy_j.csv --dataset_p standardDs.zip --ori_eval_p newLab/lab0/figs/eval.json --outdir newLab/lab0/feedbackC --unlabelled_p unlabelled.json
