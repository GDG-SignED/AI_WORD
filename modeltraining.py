from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# ✅ 최적 모델 저장 경로
model_save_path = "./Trained_model"
os.makedirs(model_save_path, exist_ok=True)

# ✅ 데이터 로드
data_path = "./Processed_dataset"
X_train = np.load(os.path.join(data_path, "X_train.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))
X_val = np.load(os.path.join(data_path, "X_val.npy"))
y_val = np.load(os.path.join(data_path, "y_val.npy"))

# ✅ Transformer 모델 정의 (과적합 방지 적용)
def build_transformer_model(input_shape, num_classes, d_model=128, num_heads=4, ff_dim=256):
    inputs = Input(shape=input_shape)

    # 🔥 Transformer 인코더 블록 (L2 Regularization 추가)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.5)(x)  # 🔥 Dropout 증가

    x = Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)  # 🔥 Dropout 증가
    x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)  # 🔥 Dropout 증가
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# ✅ 모델 빌드
input_shape = (30, 126)
num_classes = y_train.shape[1]
model = build_transformer_model(input_shape, num_classes)

# ✅ 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # 🔥 Label Smoothing 추가
    metrics=["accuracy"]
)

# ✅ Early Stopping & Model Checkpoint 설정 (patience 조정)
best_model_path = os.path.join(model_save_path, "./trained_model/best_transformer_model.keras")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1)  # 🔥 학습률 감소 비율 조정
]

# ✅ 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks
)

# ✅ Best 모델 로드
model.load_weights(best_model_path)

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

# ✅ 학습 그래프 출력
plot_training_history(history)

# ✅ 데이터 및 샘플 출력
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print("샘플 데이터:\n", X_train[0])  # 첫 번째 샘플 출력
print("샘플 라벨:\n", y_train[0])  # 첫 번째 샘플의 라벨 출력
