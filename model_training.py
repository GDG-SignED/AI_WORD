import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# 최적 모델 저장 경로 
model_save_path = "./Model/"

# ✅ 데이터 로드
data_path = "./AI_WORD/processed_dataset"
X_train = np.load(os.path.join(data_path, "X_train.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))
X_val = np.load(os.path.join(data_path, "X_val.npy"))
y_val = np.load(os.path.join(data_path, "y_val.npy"))

# 데이터 Shape 확인
print(f"✅ X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"✅ X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# ✅ 모델 정의 (Transformer 기반)
def build_transformer_model(input_shape, num_classes, d_model=128, num_heads=4, ff_dim=256):
    inputs = Input(shape=input_shape)

    # Transformer 인코더 블록
    x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    x = Flatten()(x)  # 시퀀스 데이터를 펼쳐서 FC layer에 입력
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# 모델 빌드
input_shape = (30, 126)  # 30 프레임, 126 특징 (양손)
num_classes = y_train.shape[1]  # 원-핫 인코딩된 클래스 수
model = build_transformer_model(input_shape, num_classes)

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Early Stopping & Model Checkpoint 설정
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        "./AI_WORD/trained_model/best_transformer_model.keras", 
        monitor="val_loss", 
        save_best_only=True, 
        verbose=1
    )
]

# ✅ 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,  # 🔥 200 에포크
    batch_size=32,
    callbacks=callbacks
)

# ✅ 학습 그래프 그리기
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # 🔹 Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # 🔹 Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")

    plt.show()

# 학습 그래프 출력
plot_training_history(history)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print("샘플 데이터:\n", X_train[0])  # 첫 번째 샘플 출력
print("샘플 라벨:\n", y_train[0])  # 첫 번째 샘플의 라벨 출력
