import fire
from dataclasses import dataclass


@dataclass
class Param:
    x_train: str
    y_train: str
    x_test: str
    y_test: str
    max_len: int
    upsumpling: int


if __name__ == "__main__":
    print(fire.Fire(Param))
