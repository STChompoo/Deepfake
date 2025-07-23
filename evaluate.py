import os
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from app.predictor import classify_image

# กำหนด path ไปยังโฟลเดอร์ที่มีข้อมูลทดสอบ
TEST_DATASET_PATH = "test_dataset"  # สมมติว่าเป็น test_dataset/real, test_dataset/fake

true_labels = []
pred_labels = []

# วนลูปทุกคลาส (real, fake)
for label in os.listdir(TEST_DATASET_PATH):
    class_dir = os.path.join(TEST_DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue
    
    for filename in os.listdir(class_dir):
        file_path = os.path.join(class_dir, filename)

        try:
            image = Image.open(file_path).convert("RGB")
            np_image = np.array(image)

            prediction_result = classify_image(np_image)
            predicted_label = prediction_result["prediction"].lower()

            true_labels.append(label.lower())
            pred_labels.append(predicted_label)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# แสดงผล metrics
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=["real", "fake"]))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels, labels=["real", "fake"]))
