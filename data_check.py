import numpy as np

X_train = np.load("./AI_WORD/processed_dataset/X_train.npy")
y_train = np.load("./AI_WORD/processed_dataset/y_train.npy")
X_val = np.load("./AI_WORD/processed_dataset/X_val.npy")
y_val = np.load("./AI_WORD/processed_dataset/y_val.npy")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# 첫 번째 샘플 데이터 확인
print("첫 번째 샘플 X_train[0]:\n", X_train[0])
print("첫 번째 샘플 라벨 y_train[0]:", y_train[0])
