import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.keras"
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    model = tf.keras.models.load_model(model_path)

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.squeeze()

    probs = model.predict(x_test, batch_size=128, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nMODEL:", model_path)
    print("\nCLASSIFICATION REPORT:\n")
    print(report)

    os.makedirs("artifacts", exist_ok=True)
    with open(f"artifacts/classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (CIFAR-10) - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(10), CLASS_NAMES)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"artifacts/confusion_matrix_{model_name}.png", dpi=200)

    print(f"\nSaved artifacts/confusion_matrix_{model_name}.png and artifacts/classification_report_{model_name}.txt")

if __name__ == "__main__":
    main()
