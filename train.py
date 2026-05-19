import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# For repeatability
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def save_learning_curves(history: keras.callbacks.History) -> None:
    os.makedirs("artifacts", exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig("artifacts/accuracy_curve.png", dpi=200)
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig("artifacts/loss_curve.png", dpi=200)
    plt.close()


def build_model() -> keras.Model:
    # Simple but strong CNN + BatchNorm + Dropout
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Data augmentation (only active during training)
    # x = layers.RandomFlip("horizontal")(x)
    # x = layers.RandomTranslation(0.1, 0.1)(x)
    # x = layers.RandomZoom(0.1)(x)

    def conv_block(tensor: tf.Tensor, filters: int) -> tf.Tensor:
        tensor = layers.Conv2D(filters, 3, padding="same")(tensor)
        tensor = layers.BatchNormalization()(tensor)
        tensor = layers.Activation("relu")(tensor)

        tensor = layers.Conv2D(filters, 3, padding="same")(tensor)
        tensor = layers.BatchNormalization()(tensor)
        tensor = layers.Activation("relu")(tensor)

        tensor = layers.MaxPooling2D()(tensor)
        tensor = layers.Dropout(0.25)(tensor)
        return tensor

    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar10_cnn")
    return model


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Split train into train/val
    val_size = 5000
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="model.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    save_learning_curves(history)
    print("Saved artifacts/accuracy_curve.png and artifacts/loss_curve.png")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save class names for deployment
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/classes.txt", "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")

    print("Saved best model to model.keras and classes to artifacts/classes.txt")


if __name__ == "__main__":
    main()
