import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def save_learning_curves(history, tag):
    os.makedirs("artifacts", exist_ok=True)

    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"Accuracy (Train vs Val) - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(f"artifacts/accuracy_curve_{tag}.png", dpi=200)

    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"Loss (Train vs Val) - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(f"artifacts/loss_curve_{tag}.png", dpi=200)

def build_transfer_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Rescaling(1.0/255)(inputs)

    # Upscale to what MobileNetV2 expects
    x = layers.Resizing(96, 96)(x)

    # Augmentation
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    val_size = 5000
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    model, base = build_transfer_model()

    # Phase 1: train head
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        keras.callbacks.ModelCheckpoint("model_transfer.keras", monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    h1 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=cb,
        verbose=1
    )
    save_learning_curves(h1, "transfer_head")

    # Phase 2: fine-tune top of backbone
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    h2 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=cb,
        verbose=1
    )
    save_learning_curves(h2, "transfer_finetune")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTransfer Test accuracy: {test_acc:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/classes.txt", "w") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")

    print("Saved model_transfer.keras and updated artifacts/classes.txt")

if __name__ == "__main__":
    main()
