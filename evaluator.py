import os
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from app.predictor import classify_image

def evaluate_model_on_folder(real_path, fake_path):
    true_labels = []
    pred_labels = []

    for label, path in [("Real", real_path), ("Fake", fake_path)]:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            try:
                image = Image.open(filepath).convert("RGB")
                np_image = np.array(image)
                result = classify_image(np_image)
                predicted = result["prediction"].capitalize()
                true_labels.append(label)
                pred_labels.append(predicted)
            except Exception as e:
                print(f"Error with {filepath}: {e}")

    # Generate report
    report = classification_report(true_labels, pred_labels, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)
    report["accuracy"] = accuracy

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=["Real", "Fake"])
    cm_img = _plot_confusion_matrix(cm, labels=["Real", "Fake"])

    return report, cm_img

def _plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
