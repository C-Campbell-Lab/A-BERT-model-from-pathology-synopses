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
    transformers = "^3.3.1"
    torch = "^1.6.0"
    scikit-learn = "^0.23.2"
    tqdm = "^4.50.1"
    dash = "^1.16.2"
    requests = "^2.24.0"
    plotly = "^4.11.0"
    wheel = "^0.35.1"
    fire = "^0.3.1"
    kaleido = "^0.1.0"


## Installation guide

    !pip -q install tagc --upgrade
    !pip -q install -U kaleido

*It takes about 5-10 mins.*

## Demo

> The data that support the findings of this study are available on reasonable request from the corresponding author, pending local REB and privacy office approval. The data are not publicly available because they contain information that could compromise research participant privacy/consent.

**You need first to contact the corresponding author to get the dataset "stdDs.zip" and "unlabelled.json"**

### Colab

**https://colab.research.google.com/drive/1wLzsaWWgPkoRgrUsL94Iv8_FiSiA7sKF?usp=sharing**


### Results

You can check the results in the Colab notebook above. The running time is about 10 mins.

## Instructions for use

We recommend you follow the instructions in the Colab notebook above.
