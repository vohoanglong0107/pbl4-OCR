import argparse
import os
import pandas as pd


def rename_file_dataset(data_dir):
    data = pd.DataFrame(
        columns=["filename", "words"], index=pd.RangeIndex(start=0, stop=1000, step=1)
    )
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                row = filename.split("_")
                assert len(row) == 2
                words = row[0]
                fileindex = row[1]
                data.loc[int(fileindex[:-4])] = [fileindex, words]
                os.rename(
                    os.path.join(dirpath, filename), os.path.join(dirpath, fileindex)
                )
    data.to_csv(os.path.join(data_dir, "labels.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/en/train/")
    args = parser.parse_args()
    rename_file_dataset(args.data_dir)
