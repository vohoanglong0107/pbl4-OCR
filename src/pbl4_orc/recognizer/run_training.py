import argparse
import os
from typing import Optional

import pandas as pd
import yaml

from .train import train


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)

    if opt.lang_char == "None":
        characters = ""
        for data in opt["select_data"].split("-"):
            csv_path = os.path.join(opt["train_data"], data, "labels.csv")
            df = pd.read_csv(
                csv_path,
                sep="^([^,]+),",
                engine="python",
                usecols=["filename", "words"],
                keep_default_na=False,
            )
            all_char = "".join(df["words"])
            characters += "".join(set(all_char))
        characters = sorted(set(characters))
        opt.character = "".join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)
    return opt


def main(sys_args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/en_config.yaml")
    args = parser.parse_args(sys_args)
    opt = get_config(args.config)
    train(opt)


if __name__ == "__main__":
    main()
