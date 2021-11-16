import os

import numpy as np
import pytest

from pbl4_ocr import __version__
from pbl4_ocr import Reader
from pbl4_ocr.utils import get_raw_text


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
        "Screenshot 2021-11-08 211343.png",
        "123.jpg",
    ],
)
def test_predict_custom_example(image):
    reader = Reader(["en"], recog_network="custom_example")

    print(reader.readtext(os.path.join("image", image)))


@pytest.mark.parametrize(
    "image",
    [
        "102022304261145309001.jpg",
        "Screenshot 2021-11-04 153513.png",
        "Screenshot 2021-11-04 154451.png",
        "Screenshot 2021-11-05 205551.png",
        "Screenshot 2021-11-05 213832.png",
        "Screenshot 2021-11-08 211343.png",
        "123.jpg",
    ],
)
def test_predict_TRBA(image):
    reader = Reader(["en"], recog_network="TRBA")

    print(reader.readtext(os.path.join("image", image)))


@pytest.mark.parametrize(
    "image",
    [
        "102022304261145309001.jpg",
        "Screenshot 2021-11-04 153513.png",
        "Screenshot 2021-11-04 154451.png",
        "Screenshot 2021-11-05 205551.png",
        "Screenshot 2021-11-05 213832.png",
        "Screenshot 2021-11-08 211343.png",
        "123.jpg",
        "935.jpg",
    ],
)
def test_predict_VBC(image):
    reader = Reader(["en"], recog_network="VBC")

    print(get_raw_text(reader.readtext(os.path.join("image", image))))

