#!/usr/bin/env python3
"""
Plot training and validation loss from CSV log.
Usage: python scripts/plot_training.py [path_to_csv]
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


def plot_training(csv_path="checkpoints/training_log.csv"):
    """Plot training curves from CSV log."""
    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        print("Make sure training has started or provide correct path")
        sys.exit(1)

    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            learning_rates.append(float(row["learning_rate"]))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot losses
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Highlight best validation loss
    min_val_idx = val_losses.index(min(val_losses))
    ax1.plot(
        epochs[min_val_idx],
        val_losses[min_val_idx],
        "g*",
        markersize=15,
        label=f"Best Val: {val_losses[min_val_idx]:.4f}",
    )
    ax1.legend()

    # Plot learning rate
    ax2.plot(epochs, learning_rates, "g-", label="Learning Rate", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = csv_file.parent / "training_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary
    print("\nTraining Summary:")
    print(f"  Total epochs: {len(epochs)}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Best val loss: {min(val_losses):.4f} (epoch {epochs[min_val_idx]})")

    plt.show()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/training_log.csv"
    plot_training(csv_path)
