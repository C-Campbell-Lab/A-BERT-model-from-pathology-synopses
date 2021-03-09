from tagc.data_utils import rawdata_stat
from tagc.io_utils import load_datazip
from tagc.visualization import plot_tag_stat
from pathlib import Path


def plot_ds(dsp):
    dataset = load_datazip(dsp)
    tag_stat = rawdata_stat(dataset)
    fig = plot_tag_stat(tag_stat)
    fig.write_image(str(Path(dsp).with_suffix(".pdf")))


if __name__ == "__main__":
    from fire import Fire

    Fire(plot_ds)
