import os
import sys
import argparse
import tensorflow as tf

from model import build_model, verify_conv_layers
from data_loader import download_dataset, load_datasets, DEFAULT_IMAGE_SIZE
import train as train_module
import evaluate as evaluate_module
import predict as predict_module


def cmd_verify(args):
    print("=" * 60)
    print("STEP 1: ENVIRONMENT & ARCHITECTURE VERIFICATION")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available     : {len(tf.config.list_physical_devices('GPU')) > 0}")

    print("\n[Architecture Check] Building CNN model to inspect Conv2D layers...")
    model = build_model(input_shape=(128, 128, 3))
    model.summary()

    is_valid = verify_conv_layers(model)
    if is_valid:
        print("\n[PASSED] Architecture strictly satisfies requirement: EXACTLY 3 Conv2D layers.")
    else:
        print("\n[FAILED] Architecture does not have exactly 3 Conv2D layers!")
        sys.exit(1)


def cmd_prepare_data(args):
    print("=" * 60)
    print("STEP 2: DATA PREPARATION & SPLITTING")
    print("=" * 60)
    dataset_dir = download_dataset(dest_dir=args.data_dir)
    print("\nVerifying train/val/test splits...")
    train_ds, val_ds, test_ds, classes = load_datasets(
        data_dir=dataset_dir,
        batch_size=args.batch_size
    )
    print(f"[Done] Datasets ready. Class labels: {classes}")


def cmd_train(args):
    train_module.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_save_path=args.model_path,
        data_dir=args.data_dir
    )


def cmd_evaluate(args):
    evaluate_module.evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )


def cmd_predict(args):
    img_path = args.image_path
    if not img_path:
        img_path = predict_module.find_sample_image(data_dir=args.data_dir)
        if not img_path:
            print("[Error] No image provided and no sample images found in dataset.")
            print("Please specify an image with: python main.py predict --image-path path/to/image.jpg")
            return

    predict_module.predict_single_image(
        image_path=img_path,
        model_path=args.model_path,
        output_vis_path=args.output
    )


def cmd_full_run(args):
    print("=" * 70)
    print("EXECUTING FULL END-TO-END PIPELINE")
    print("=" * 70)
    cmd_verify(args)
    cmd_prepare_data(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_predict(args)
    print("=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Dog vs Cat 3-Layer CNN Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_verify = subparsers.add_parser("verify", help="Check dependencies and 3-Conv2D model structure")
    p_verify.set_defaults(func=cmd_verify)

    p_prep = subparsers.add_parser("prepare-data", help="Download and organize dataset")
    p_prep.add_argument("--data-dir", type=str, default="data", help="Directory to store dataset")
    p_prep.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_prep.set_defaults(func=cmd_prepare_data)

    p_train = subparsers.add_parser("train", help="Train the CNN model")
    p_train.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_train.add_argument("--data-dir", type=str, default="data/cats_and_dogs_filtered", help="Dataset path")
    p_train.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Model save path")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model on test data")
    p_eval.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Model path")
    p_eval.add_argument("--data-dir", type=str, default="data/cats_and_dogs_filtered", help="Dataset path")
    p_eval.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = subparsers.add_parser("predict", help="Predict class of an image")
    p_pred.add_argument("--image-path", type=str, default=None, help="Path to image file")
    p_pred.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Model path")
    p_pred.add_argument("--data-dir", type=str, default="data", help="Data directory")
    p_pred.add_argument("--output", type=str, default="results/prediction_sample.png", help="Output preview path")
    p_pred.set_defaults(func=cmd_predict)

    p_full = subparsers.add_parser("full-run", help="Run entire pipeline: verify, download, train, evaluate, predict")
    p_full.add_argument("--epochs", type=int, default=15, help="Epochs")
    p_full.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_full.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_full.add_argument("--data-dir", type=str, default="data/cats_and_dogs_filtered", help="Dataset directory")
    p_full.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Model path")
    p_full.add_argument("--image-path", type=str, default=None, help="Sample image path")
    p_full.add_argument("--output", type=str, default="results/prediction_sample.png", help="Output preview path")
    p_full.set_defaults(func=cmd_full_run)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
