import os
import shutil
import zipfile
import urllib.request
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32


def download_dataset(
    dest_dir: str = "data",
    url: str = DATASET_URL
) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    extracted_path = os.path.join(dest_dir, "cats_and_dogs_filtered")

    train_dir = os.path.join(extracted_path, "train")
    validation_dir = os.path.join(extracted_path, "validation")
    if os.path.exists(train_dir) and os.path.exists(validation_dir):
        print(f"[Dataset] Verified existing dataset at: {extracted_path}")
        return extracted_path

    zip_path = os.path.join(dest_dir, "cats_and_dogs_filtered.zip")
    print(f"[Dataset] Downloading Cats and Dogs dataset from:\n  {url}")

    try:
        archive_path = tf.keras.utils.get_file(
            fname="cats_and_dogs_filtered.zip",
            origin=url,
            extract=False,
            cache_dir=os.path.abspath(dest_dir)
        )
        zip_path = archive_path
    except Exception as e:
        print(f"[Warning] tf.keras.utils.get_file failed ({e}). Falling back to urllib...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as net_err:
            raise RuntimeError(
                f"Failed to download dataset. Check your internet connection or URL: {net_err}"
            ) from net_err

    print(f"[Dataset] Extracting archive {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    except Exception as err:
        raise RuntimeError(f"Corrupted archive extraction failed: {err}") from err

    print(f"[Dataset] Extracted successfully to: {extracted_path}")
    return extracted_path


def get_data_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation"
    )


def load_datasets(
    data_dir: Optional[str] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augment: bool = True,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    if data_dir is None or not os.path.exists(data_dir):
        data_dir = download_dataset(dest_dir="data")

    train_folder = os.path.join(data_dir, "train")
    val_test_folder = os.path.join(data_dir, "validation")

    if not os.path.exists(train_folder) or not os.path.exists(val_test_folder):
        raise FileNotFoundError(
            f"Expected 'train' and 'validation' directories inside: {data_dir}"
        )

    print(f"[DataLoader] Loading Training set from: {train_folder}")
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_folder,
        labels="inferred",
        label_mode="binary",
        class_names=["cats", "dogs"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    class_names = raw_train_ds.class_names
    print(f"[DataLoader] Detected classes: {class_names} (0 = {class_names[0]}, 1 = {class_names[1]})")

    print(f"[DataLoader] Loading Validation & Test set from: {val_test_folder}")
    raw_val_test_ds = tf.keras.utils.image_dataset_from_directory(
        val_test_folder,
        labels="inferred",
        label_mode="binary",
        class_names=["cats", "dogs"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    total_val_batches = tf.data.experimental.cardinality(raw_val_test_ds).numpy()
    val_batches = total_val_batches // 2

    raw_val_ds = raw_val_test_ds.take(val_batches)
    raw_test_ds = raw_val_test_ds.skip(val_batches)

    print(f"[DataLoader] Batch counts -> Train: {tf.data.experimental.cardinality(raw_train_ds).numpy()} | "
          f"Validation: {val_batches} | Test: {total_val_batches - val_batches}")

    rescale_layer = layers.Rescaling(1.0 / 255.0, name="rescaling_norm")

    aug_pipeline = get_data_augmentation() if augment else None

    def prep_train(img, lbl):
        if aug_pipeline is not None:
            img = aug_pipeline(img, training=True)
        img = rescale_layer(img)
        return img, lbl

    def prep_eval(img, lbl):
        img = rescale_layer(img)
        return img, lbl

    autotune = tf.data.AUTOTUNE

    train_ds = (
        raw_train_ds
        .map(prep_train, num_parallel_calls=autotune)
        .cache()
        .prefetch(buffer_size=autotune)
    )

    val_ds = (
        raw_val_ds
        .map(prep_eval, num_parallel_calls=autotune)
        .cache()
        .prefetch(buffer_size=autotune)
    )

    test_ds = (
        raw_test_ds
        .map(prep_eval, num_parallel_calls=autotune)
        .cache()
        .prefetch(buffer_size=autotune)
    )

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    print("Testing data loader module...")
    try:
        tr, va, te, classes = load_datasets()
        print(f"Dataset successfully initialized. Classes: {classes}")
        for imgs, lbls in tr.take(1):
            print(f"Train batch image shape: {imgs.shape}, labels shape: {lbls.shape}")
            print(f"Pixel min: {tf.reduce_min(imgs):.3f}, max: {tf.reduce_max(imgs):.3f}")
            break
    except Exception as e:
        print(f"Data loader test notice: {e}")
