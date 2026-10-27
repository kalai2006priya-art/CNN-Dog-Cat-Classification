import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_model(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    learning_rate: float = 0.001
) -> keras.Model:
    model = models.Sequential(name="Cat_Dog_CNN_3Layers")

    model.add(layers.Input(shape=input_shape, name="input_layer"))

    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_layer_1"
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"))

    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_layer_2"
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"))

    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_layer_3"
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_3"))

    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(128, activation="relu", name="dense_features"))
    model.add(layers.Dropout(0.5, name="dropout_regularization"))

    model.add(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))

    conv_count = count_conv_layers(model)
    if conv_count != 3:
        raise ValueError(
            f"Architecture constraint violation: Expected exactly 3 Conv2D layers, but found {conv_count}."
        )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def count_conv_layers(model: keras.Model) -> int:
    return sum(isinstance(layer, layers.Conv2D) for layer in model.layers)


def verify_conv_layers(model: keras.Model) -> bool:
    count = count_conv_layers(model)
    print(f"[Verification] Number of Conv2D layers detected: {count}")
    return count == 3


if __name__ == "__main__":
    print("Testing CNN Model creation...")
    test_model = build_model(input_shape=(128, 128, 3), learning_rate=0.001)
    test_model.summary()
    assert verify_conv_layers(test_model), "Model does NOT have exactly 3 Conv2D layers!"
    print("\nSUCCESS: Model successfully built and verified with exactly 3 Conv2D layers.")
