"""Step 6: plot loss and accuracy curves from outputs/training_history.json.

Saves: outputs/plots/training_curves.png

How to read the plot:
    - train curve good, val curve bad  -> overfitting (model memorizes)
    - both curves bad                  -> underfitting (model too weak / too little training)

Usage:
    python src/plot_history.py
"""

import matplotlib

matplotlib.use("Agg")   # draw to a file, no window needed
import matplotlib.pyplot as plt

from utils import HISTORY_PATH, PLOTS_DIR, load_json

PLOT_PATH = PLOTS_DIR / "training_curves.png"


def plot_metric(axis, history, train_key, val_key, title, ylabel):
    """Draw train and validation lines of one metric on one axis."""
    epochs = range(1, len(history[train_key]) + 1)
    axis.plot(epochs, history[train_key], marker="o", label="train")
    axis.plot(epochs, history[val_key], marker="o", label="val")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"{HISTORY_PATH} not found. Run train.py first.")

    history = load_json(HISTORY_PATH)

    figure, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(11, 4))
    plot_metric(loss_axis, history, "train_loss", "val_loss", "Loss", "Loss")
    plot_metric(acc_axis, history, "train_acc", "val_acc", "Accuracy", "Accuracy")
    figure.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=150)
    print("Saved plot to:", PLOT_PATH)


if __name__ == "__main__":
    main()
