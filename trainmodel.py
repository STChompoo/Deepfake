import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.xception import Xception, preprocess_input
from keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(real_dir, fake_dir, img_size=(224, 224), max_images_per_class=500):
    X, y = [], []

    for label, folder in enumerate([real_dir, fake_dir]):  # 0 = real, 1 = fake
        count = 0
        for fname in os.listdir(folder):
            if count >= max_images_per_class:
                break
            fpath = os.path.join(folder, fname)
            try:
                img = load_img(fpath, target_size=img_size)
                img = img_to_array(img)
                img = preprocess_input(img)
                X.append(img)
                y.append(label)
                count += 1
            except Exception as e:
                print(f"โหลด {fpath} ไม่ได้: {e}")

    print(f"โหลดรูปทั้งหมดสำเร็จ: {len(X)} รูป")
    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)


def build_model(input_shape=(224, 224, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # ไม่เทรน base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # ใช้ sigmoid เพราะ binary classification
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(real_dir, fake_dir, save_path="model/deepfake_xception.h5"):
    # โหลดข้อมูล
    X_train, X_val, y_train, y_val = load_dataset(real_dir, fake_dir, max_images_per_class=500)

    # สร้างโมเดล
    model = build_model()

    # สร้างโฟลเดอร์ model ถ้ายังไม่มี
    os.makedirs("model", exist_ok=True)

    # ตั้งค่าให้เซฟเฉพาะโมเดลที่แม่นยำที่สุด
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, mode='max')

    # ใช้ Data Augmentation เพิ่มความหลากหลาย
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow(X_train, y_train, batch_size=32)

    # เทรนโมเดล
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=10,
        steps_per_epoch=len(X_train) // 32,
        callbacks=[checkpoint]
    )

    return model, history


if __name__ == "__main__":
    real_path = "frames/Real"
    fake_path = "frames/Fake"
    model, history = train_model(real_path, fake_path)
