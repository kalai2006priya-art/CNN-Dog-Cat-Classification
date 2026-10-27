import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from model import build_model, verify_conv_layers
from data_loader import load_datasets, DEFAULT_IMAGE_SIZE, DEFAULT_BATCH_SIZE


def plot_and_save_history(history: keras.callbacks.History, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, label="Training Accuracy", color="#2b5c8f", marker="o", linewidth=2)
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#d95f02", marker="s", linewidth=2)
    axes[0].set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    axes[1].plot(epochs_range, loss, label="Training Loss", color="#2b5c8f", marker="o", linewidth=2)
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#d95f02", marker="s", linewidth=2)
    axes[1].set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Saved training curves plot to: {save_path}")


def train(
    epochs: int = 15,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 0.001,
    model_save_path: str = os.path.join("models", "dog_cat_cnn_model.keras"),
    results_dir: str = "results",
    data_dir: str = os.path.join("data", "cats_and_dogs_filtered")
) -> keras.callbacks.History:
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING MODEL TRAINING: Cat vs Dog 3-Layer CNN")
    print("=" * 70)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=data_dir,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=batch_size,
        augment=True
    )

    model = build_model(
        input_shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3),
        learning_rate=learning_rate
    )
    model.summary()

    if not verify_conv_layers(model):
        raise RuntimeError("Architecture check failed: model does not have exactly 3 Conv2D layers.")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks_list = [early_stopping, checkpoint, reduce_lr]

    print(f"\n[Training] Fitting model for up to {epochs} epochs (Batch size: {batch_size})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    if not os.path.exists(model_save_path):
        model.save(model_save_path)
        print(f"[Training] Saved model to: {model_save_path}")

    plot_path = os.path.join(results_dir, "training_history.png")
    plot_and_save_history(history, plot_path)

    summary_path = os.path.join(results_dir, "training_summary.json")
    final_metrics = {
        "status": "COMPLETED",
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "class_mapping": {class_names[0]: 0, class_names[1]: 1},
        "model_path": model_save_path,
        "history_plot": plot_path
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"[Training] Saved training summary to: {summary_path}")

    print("=" * 70)
    print(f"TRAINING COMPLETE! Best Validation Accuracy: {final_metrics['best_val_accuracy'] * 100:.2f}%")
    print("=" * 70)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3-Layer CNN Dog vs Cat Classifier")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data/cats_and_dogs_filtered", help="Dataset directory")
    parser.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Model save path")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_save_path=args.model_path,
        data_dir=args.data_dir
    )
