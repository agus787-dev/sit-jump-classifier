"""Step 5: train the video classifier.

Saves:
    outputs/models/best_video_model.pth   (weights with the best validation accuracy)
    outputs/models/class_names.json       (needed later by evaluate.py / predict.py)
    outputs/training_history.json         (loss and accuracy per epoch, for plotting)

Usage:
    python src/train.py
    python src/train.py --epochs 10 --batch_size 4 --learning_rate 0.0005
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders, print_batch_info
from model import SimpleVideoCNN, print_model_summary
from utils import (
    CLASS_NAMES_PATH,
    HISTORY_PATH,
    MODEL_PATH,
    MODELS_DIR,
    get_device,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple video classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate.")
    return parser.parse_args()


def run_one_epoch(model, dataloader, criterion, optimizer, device, training: bool):
    """Run one pass over the data. Returns (average loss, accuracy).

    training=True  -> weights are updated
    training=False -> evaluation only (no gradients, no updates)
    """
    model.train() if training else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.enable_grad() if training else torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def create_empty_history():
    return {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


def update_history(history, train_loss, train_acc, val_loss, val_acc):
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)


def print_epoch_summary(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
    print(
        f"Epoch [{epoch}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )


def main():
    args = parse_args()

    set_seed()
    device = get_device()
    print("Using device:", device)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, _, class_names = create_dataloaders(batch_size=args.batch_size)
    print_batch_info(train_loader, class_names)
    save_json(class_names, CLASS_NAMES_PATH)

    # Model, loss, optimizer
    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    print_model_summary(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    history = create_empty_history()
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        update_history(history, train_loss, train_acc, val_loss, val_acc)
        print_epoch_summary(epoch, args.epochs, train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best model to {MODEL_PATH}")

    save_json(history, HISTORY_PATH)

    print("\nTraining complete.")
    print("Best validation accuracy:", round(best_val_acc, 4))
    print("Saved model to:", MODEL_PATH)
    print("Saved class names to:", CLASS_NAMES_PATH)
    print("Saved history to:", HISTORY_PATH)


if __name__ == "__main__":
    main()
