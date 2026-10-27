import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_loader import DEFAULT_IMAGE_SIZE


def preprocess_image(
    image_path: str,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    img = keras.utils.load_img(image_path, target_size=target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict_single_image(
    image_path: str,
    model_path: str = os.path.join("models", "dog_cat_cnn_model.keras"),
    save_visualization: bool = True,
    output_vis_path: str = os.path.join("results", "prediction_sample.png")
) -> dict:
    if not os.path.exists(model_path):
        msg = (
            f"[PENDING RESULTS] Trained model '{model_path}' not found.\n"
            "Cannot run prediction until the model is trained.\n"
            "Train the model first using:\n"
            "    python train.py\n"
        )
        print(msg)
        return {"status": "PENDING", "message": msg}

    print(f"[Inference] Loading model: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"[Inference] Preprocessing image: {image_path}")
    img_batch = preprocess_image(image_path, target_size=DEFAULT_IMAGE_SIZE)

    prob = float(model.predict(img_batch, verbose=0)[0][0])

    if prob >= 0.5:
        predicted_class = "Dog"
        confidence = prob * 100.0
    else:
        predicted_class = "Cat"
        confidence = (1.0 - prob) * 100.0

    result = {
        "status": "SUCCESS",
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "raw_probability": round(prob, 4),
        "class_mapping": "0 = Cat, 1 = Dog"
    }

    print("-" * 50)
    print(f"PREDICTION RESULT : {predicted_class.upper()}")
    print(f"Confidence        : {confidence:.2f}%")
    print(f"Raw Sigmoid Output: {prob:.4f}")
    print("-" * 50)

    if save_visualization:
        os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
        raw_img = keras.utils.load_img(image_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(raw_img)
        title_color = "#1f77b4" if predicted_class == "Cat" else "#ff7f0e"
        plt.title(
            f"Prediction: {predicted_class} ({confidence:.1f}%)\nFile: {os.path.basename(image_path)}",
            fontsize=12,
            fontweight="bold",
            color=title_color
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_vis_path, dpi=300)
        plt.close()
        print(f"[Inference] Saved annotated prediction preview: {output_vis_path}")

    return result


def find_sample_image(data_dir: str = "data") -> str | None:
    possible_dirs = [
        os.path.join(data_dir, "sample_images"),
        os.path.join(data_dir, "cats_and_dogs_filtered", "validation", "cats"),
        os.path.join(data_dir, "cats_and_dogs_filtered", "validation", "dogs"),
        os.path.join(data_dir, "cats_and_dogs_filtered", "train", "cats")
    ]
    for d in possible_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                return os.path.join(d, files[0])
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cat or Dog for an Image")
    parser.add_argument("--image-path", type=str, default=None, help="Path to target image file")
    parser.add_argument("--model-path", type=str, default="models/dog_cat_cnn_model.keras", help="Path to trained model")
    parser.add_argument("--output", type=str, default="results/prediction_sample.png", help="Path to save result image")
    args = parser.parse_args()

    target_img = args.image_path
    if not target_img:
        target_img = find_sample_image()
        if target_img:
            print(f"[Notice] No image provided. Using detected sample image: {target_img}")
        else:
            print("[Warning] No image provided and no sample image found.")
            print("Please provide an image using: python predict.py --image-path path/to/image.jpg")
            exit(0)

    predict_single_image(
        image_path=target_img,
        model_path=args.model_path,
        output_vis_path=args.output
    )
