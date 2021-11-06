import os

import pytest

from pbl4_orc import __version__
from pbl4_orc import Reader


def test_version():
    assert __version__ == "0.1.0"


def test_load_custom_example():
    Reader(["en"], recog_network="custom_example")


@pytest.mark.parametrize(
    "image",
    [
        "102022304261145309001.jpg",
        "Screenshot 2021-11-04 153513.png",
        "Screenshot 2021-11-04 154451.png",
        "Screenshot 2021-11-05 205551.png",
        "Screenshot 2021-11-05 213832.png",
    ],
)
def test_predict_custom_example(image):
    reader = Reader(["en"], recog_network="custom_example")

    print(reader.readtext(os.path.join("image", image)))
