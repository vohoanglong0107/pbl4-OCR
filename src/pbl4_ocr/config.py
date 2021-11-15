import os

os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = (
    os.environ.get("EASYOCR_MODULE_PATH")
    or os.environ.get("MODULE_PATH")
    or os.path.expanduser("~/.EasyOCR/")
)

# detector parameters
detection_models = {
    "craft": {
        "filename": "craft_mlt_25k.pth",
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",  # noqa: E501
        "filesize": "2f8227d2def4037cdb3b34389dcf9ec1",
    }
}

# recognizer parameters
imgH = 64
separator_list = {"th": ["\xa2", "\xa3"], "en": ["\xa4", "\xa5"]}
separator_char = []
for lang, sep in separator_list.items():
    separator_char += sep
