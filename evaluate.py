import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras

from data_loader import load_datasets, DEFAULT_IMAGE_SIZE, DEFAULT_BATCH_SIZE


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix - Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix heatmap to: {save_path}")


def evaluate(
    model_path: str = os.path.join("models", "dog_cat_cnn_model.keras"),
    data_dir: str = os.path.join("data", "cats_and_dogs_filtered"),
    results_dir: str = "results",
    batch_size: int = DEFAULT_BATCH_SIZE
) -> dict:
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_path):
        msg = (
            f"\n[PENDING RESULTS] No trained model found at '{model_path}'.\n"
            "Evaluation cannot proceed because the model has not been trained yet.\n"
            "Please train the model first by running:\n"
            "    python train.py\n"
            "or via the unified runner:\n"
            "    python main.py train\n"
        )
        print(msg)
        return {"status": "PENDING", "message": "Model not trained yet."}

    print("=" * 70)
    print("STARTING TEST SET EVALUATION")
    print(f"Loading trained model: {model_path}")
    print("=" * 70)

    model = keras.models.load_model(model_path)

    _, _, test_ds, class_names = load_datasets(
        data_dir=data_dir,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=batch_size,
        augment=False
    )

    print("\n[Evaluation] Computing Test Loss and Accuracy...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"--> Test Loss     : {test_loss:.4f}")
    print(f"--> Test Accuracy : {test_accuracy * 100:.2f}%")

    print("\n[Evaluation] Generating predictions on all test batches...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n[Evaluation] Confusion Matrix Breakdown:")
    print(f"  * True Cats  (TN): {tn}")
    print(f"  * False Dogs (FP): {fp} (Cats misclassified as Dogs)")
    print(f"  * False Cats (FN): {fn} (Dogs misclassified as Cats)")
    print(f"  * True Dogs  (TP): {tp}")

    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    clf_report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    clf_report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    print("\n[Evaluation] Classification Report:")
    print(clf_report_str)

    report_text_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("=== 3-LAYER CNN CAT VS DOG CLASSIFIER - TEST EVALUATION REPORT ===\n\n")
        f.write(f"Model Path    : {model_path}\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN (Cats correctly identified): {tn}\n")
        f.write(f"FP (Cats predicted as Dogs)   : {fp}\n")
        f.write(f"FN (Dogs predicted as Cats)   : {fn}\n")
        f.write(f"TP (Dogs correctly identified): {tp}\n\n")
        f.write("Detailed Classification Metrics:\n")
        f.write(clf_report_str)
    print(f"[Evaluation] Saved text report to: {report_text_path}")

    metrics_json_path = os.path.join(results_dir, "evaluation_metrics.json")
    results_summary = {
        "status": "COMPLETED",
        "model_path": model_path,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": {
            "true_negative_cats": int(tn),
            "false_positive_dogs": int(fp),
            "false_negative_cats": int(fn),
            "true_positive_dogs": int(tp)
        },
        "classification_metrics": clf_report_dict,
        "confusion_matrix_plot": cm_plot_path,
        "classification_report_file": report_text_path
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
    print(f"[Evaluation] Saved JSON evaluation metrics to: {metrics_json_path}")

    print("=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3-Layer CNN Dog vs Cat Classifier")
    parser.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Model file path")
    parser.add_argument("--data-dir", type=str, default="data/cats_and_dogs_filtered", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
