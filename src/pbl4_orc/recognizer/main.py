import argparse
import sys
from typing import Optional, List

from . import convert_dataset_trdg
from . import run_training


def main(sys_args: Optional[List[str]] = None) -> None:
    if sys_args is None:
        sys_args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Train a model to recognize ORC-related text."
    )
    parser.add_argument(
        "command",
        type=str,
        choices=["convert-dataset", "train"],
        help=(
            "Subcommand to run, "
            "train to train model, "
            "convert-dataset to convert dataset from trdg to supported format"
        ),
    )
    args = parser.parse_args(sys_args[:1])
    if args.command == "convert-dataset":
        convert_dataset_trdg.main(sys_args[1:])
    elif args.command == "train":
        run_training.main(sys_args[1:])
    else:
        parser.print_help()
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
