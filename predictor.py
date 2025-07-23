import torch
import numpy as np
from PIL import Image
from app.model import model, processor
from app.config import id2label

def classify_image(np_image: np.ndarray):
    # สร้าง input tensor จากภาพ
    inputs = processor(images=Image.fromarray(np_image), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # ใช้ softmax แปลง logits เป็นความน่าจะเป็น
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()  # shape: (num_classes,)

    # ดึง label ที่โมเดลทำนาย
    predicted_class_idx = int(np.argmax(probs))
    predicted_label = id2label[str(predicted_class_idx)]

    # สร้าง dict ของ label → % เช่น { "real": 32.5, "fake": 67.5 }
    prob_percentages = {
        id2label[str(i)]: round(float(prob) * 100, 2)
        for i, prob in enumerate(probs)
    }

    return {
        "prediction": predicted_label,
        "probabilities": prob_percentages
    }
