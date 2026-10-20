"""Apply bounded checkpoint retention to an already running campaign."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_c_mlips.active_learning.training import prune_mace_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--watch-pid", type=int, required=True)
    args = parser.parse_args()
    while True:
        try:
            os.kill(args.watch_pid, 0)
        except ProcessLookupError:
            return
        for directory in args.root.rglob("checkpoints"):
            removed = prune_mace_checkpoints(directory)
            if removed:
                print(f"{directory}: removed {removed} older checkpoints", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
