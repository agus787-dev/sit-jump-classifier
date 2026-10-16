"""Step 7: evaluate the best saved model on the TEST set.

The test set was never used for training or for choosing the best epoch,
so this number is the honest estimate of how the model works on new videos.

Usage:
    python src/evaluate.py
"""

import torch
import torch.nn as nn

from dataset import create_dataloaders
from model import SimpleVideoCNN
from train import run_one_epoch
from utils import (
    CLASS_NAMES_PATH,
    MODEL_PATH,
    TEST_RESULTS_PATH,
    get_device,
    load_json,
    save_json,
    set_seed,
)


def load_trained_model(class_names, device):
    """Create the model and load the saved weights."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")

    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def collect_predictions(model, dataloader, device):
    """Return two lists: true labels and predicted labels for the whole loader."""
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences.to(device))
            predictions = torch.argmax(outputs, dim=1).cpu()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())
    return true_labels, predicted_labels


def build_confusion_matrix(true_labels, predicted_labels, num_classes):
    """matrix[true][predicted] = number of clips."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, predicted in zip(true_labels, predicted_labels):
        matrix[true][predicted] += 1
    return matrix


def print_confusion_matrix(matrix, class_names):
    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    header = " " * 12 + "".join(f"{name:>10}" for name in class_names)
    print(header)
    for name, row in zip(class_names, matrix):
        print(f"{name:>12}" + "".join(f"{value:>10}" for value in row))


def main():
    set_seed()
    device = get_device()
    print("Using device:", device)

    class_names = load_json(CLASS_NAMES_PATH)
    _, _, test_loader, _ = create_dataloaders(batch_size=2)
    model = load_trained_model(class_names, device)

    test_loss, test_acc = run_one_epoch(
        model, test_loader, nn.CrossEntropyLoss(), None, device, training=False
    )
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    true_labels, predicted_labels = collect_predictions(model, test_loader, device)
    matrix = build_confusion_matrix(true_labels, predicted_labels, len(class_names))
    print_confusion_matrix(matrix, class_names)

    save_json(
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "class_names": class_names,
            "confusion_matrix": matrix,
        },
        TEST_RESULTS_PATH,
    )
    print("\nSaved test results to:", TEST_RESULTS_PATH)


if __name__ == "__main__":
    main()
