import argparse
import sys
from typing import Optional, List
from . import recognizer


def main(sys_args: Optional[List[str]] = None) -> None:
    if sys_args is None:
        sys_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="PBL4 ORC")
    parser.add_argument(
        "command",
        type=str,
        choices=["recognize"],
        help="Subcommand to run, recognizer for recognize related command",
    )
    args = parser.parse_args(sys_args[:1])
    if args.command == "recognize":
        recognizer.main(sys_args[1:])
    else:
        parser.print_help()
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
