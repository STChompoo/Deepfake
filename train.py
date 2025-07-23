from transformers import AutoImageProcessor, SiglipForImageClassification, TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
from pathlib import Path
import torch
import os

# CONFIG
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
FEEDBACK_PATH = Path("feedback_data/")
LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}
SAVE_PATH = "./retrained_model"

# Step 1: Load feedback images
def load_images():
    image_data = {"image": [], "label": []}
    for label in LABEL2ID.keys():
        folder = FEEDBACK_PATH / label
        if not folder.exists():
            continue
        for file in folder.iterdir():
            if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                img = Image.open(file).convert("RGB")
                image_data["image"].append(img)
                image_data["label"].append(LABEL2ID[label])
            except Exception as e:
                print(f"⚠️ Skipping file {file}: {e}")
    return Dataset.from_dict(image_data)

# Step 2: Preprocessing (image -> pixel_values)
def transform(example):
    processed = processor(images=example["image"], return_tensors="pt")
    example["pixel_values"] = processed["pixel_values"][0]
    return {"pixel_values": example["pixel_values"], "labels": example["label"]}

if __name__ == "__main__":
    # Load processor & model
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    # Load and transform dataset
    dataset = load_images()
    dataset = dataset.map(transform)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=10,
        logging_steps=5,
        learning_rate=5e-5,
        save_total_limit=1,
        remove_unused_columns=False,
        logging_dir="./logs"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save retrained model
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)

    print(f"✅ Model retrained and saved at {SAVE_PATH}")
