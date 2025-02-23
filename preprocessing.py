import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ✅ 제스처 리스트
GESTURE_LIST = [
    '가족,식구,세대,가구',
    '감기',
    '건강,기력,강건하다,튼튼하다',
    '검사',
    '결혼,혼인,화혼',
    '꿈,포부,꿈꾸다',
    '남동생',
    '남편,배우자,서방',
    '낫다,치유',
    '살다,삶,생활',
    '쉬다,휴가,휴게,휴식,휴양',
    '습관,버릇',
    '신기록',
    '안녕,안부',
    '약',
    '양치질,양치',
    '어머니,모친,어미,엄마',
    '여행',
    '오빠,오라버니',
    '취미'
]

# ✅ 데이터 경로
dataset_path = "./AI_WORD/output_npy"
max_frames = 30  # 고정된 프레임 길이
feature_size = 126  # (두 손 기준: 21 랜드마크 * 3좌표 * 2손)

# ✅ 정규화 함수 (최소-최대 정규화)
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

# ✅ 시퀀스 길이 조정 함수
def normalize_sequence_length(data, target_length, feature_size):
    current_length = len(data)
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, feature_size), dtype='float32')
        return np.vstack((data, padding))
    return data[:target_length]

# ✅ 데이터 전처리 함수
def process_gesture_data(actions, dataset_path, max_frames, feature_size):
    X, y = [], []
    for idx, gesture in enumerate(actions):
        folder_path = os.path.join(dataset_path, gesture)
        if not os.path.exists(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(folder_path, file_name)
                sequence_data = np.load(file_path)

                # ✅ 시퀀스 길이 통일 (30, feature_size)
                sequence_data = normalize_sequence_length(sequence_data, max_frames, sequence_data.shape[1])

                # ✅ 한 손 동작일 경우 오른손 패딩 추가
                if sequence_data.shape[1] == 63:
                    padding = np.zeros((max_frames, 63))
                    sequence_data = np.hstack((sequence_data, padding))  # (30, 126) 변환

                # ✅ 정규화 적용
                sequence_data = normalize_data(sequence_data)

                # ✅ 원본 데이터 저장
                X.append(sequence_data)
                y.append(idx)

                # ✅ 데이터 증강 (좌우 반전)
                flipped_data = sequence_data.copy()
                flipped_data[:, 0::3] *= -1  # X 좌표 반전
                X.append(flipped_data)
                y.append(idx)

    return np.array(X, dtype='float32'), np.array(y, dtype='int')

# ✅ 데이터 저장 함수
def save_preprocessed_data(X, y, save_dir="./Processed_dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # ✅ One-Hot Encoding 적용
    y_train = to_categorical(y_train, num_classes=len(GESTURE_LIST))
    y_val = to_categorical(y_val, num_classes=len(GESTURE_LIST))

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

    print(f"✅ 데이터 저장 완료: {save_dir}")

# ✅ 실행
X_data, y_labels = process_gesture_data(GESTURE_LIST, dataset_path, max_frames, feature_size)
save_preprocessed_data(X_data, y_labels)
