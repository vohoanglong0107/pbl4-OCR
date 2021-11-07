import argparse
import os
from typing import List, Optional
import pandas as pd


def rename_file_dataset(data_dir):
    dirs = os.listdir(data_dir)
    data = pd.DataFrame(
        columns=["filename", "words"], index=pd.RangeIndex(start=0, stop=len(dirs), step=1)
    )
    for i, filename in enumerate(dirs):
        if filename.endswith(".jpg"):
            row = filename.split("_")
            assert len(row) == 2
            words = row[0]
            fileindex = str(i) + ".jpg"
            os.rename(
                os.path.join(data_dir, filename), os.path.join(data_dir, fileindex)
            )
            data.loc[int(fileindex[:-4])] = [fileindex, words]
    data.to_csv(os.path.join(data_dir, "labels.csv"), index=False)


def main(sys_args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/train/")
    args = parser.parse_args(sys_args)
    rename_file_dataset(args.data_dir)


if __name__ == "__main__":
    main()
