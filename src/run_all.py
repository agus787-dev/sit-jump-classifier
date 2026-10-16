"""Run the whole pipeline, one step after another.

Usage:
    python run_all.py                              # all steps
    python run_all.py --from-step extract_frames   # start from a later step
    python run_all.py --epochs 10 --batch_size 4   # extra options go to train.py

Steps (in order):
    download_data -> extract_videos -> extract_frames -> prepare_sequences
    -> train -> plot_history -> evaluate
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src" if (HERE / "src").is_dir() else HERE
STEPS = [
    "download_data",
    "extract_videos",
    "extract_frames",
    "prepare_sequences",
    "train",
    "plot_history",
    "evaluate",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--from-step", choices=STEPS, default=STEPS[0],
                        help="Start from this step (earlier steps are skipped).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args()


def build_train_options(args):
    """Turn the optional training arguments into command line options."""
    options = []
    for name in ("epochs", "batch_size", "learning_rate"):
        value = getattr(args, name)
        if value is not None:
            options += [f"--{name}", str(value)]
    return options


def run_step(step: str, extra_options=()):
    """Run src/<step>.py with the same Python that runs this script."""
    print(f"\n{'=' * 60}\nSTEP: {step}\n{'=' * 60}")
    command = [sys.executable, str(SRC_DIR / f"{step}.py"), *extra_options]
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    train_options = build_train_options(args)

    for step in STEPS[STEPS.index(args.from_step):]:
        run_step(step, train_options if step == "train" else ())

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
