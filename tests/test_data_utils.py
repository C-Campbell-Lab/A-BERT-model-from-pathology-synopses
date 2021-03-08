from tagc.data_utils import upsampling
from tagc.io_utils import load_datazip


def test_upsampling():
    dsp = "out/standard/standardDs0.zip"
    ds = load_datazip(dsp)
    new_x1, _ = upsampling(ds.x_train_dict, ds.y_train_tags, target=200)
    new_x2, _ = upsampling(ds.x_train_dict, ds.y_train_tags, target=-200)
    print(len(new_x1))
    assert len(new_x1) == len(new_x2)
