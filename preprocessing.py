import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 제스처 정의
gestures = [
    "가족,식구,세대,가구", "감기", "건강,기력,강건하다,튼튼하다", "검사", "결혼,혼인,화혼",
    "고모", "꿈,포부,꿈꾸다", "남동생", "남편,배우자,서방", "낫다,치유", "노래,음악,가요",
    "누나,누님", "다니다", "동생", "머무르다,존재,체류,계시다,묵다", "모자(관계)", "몸살",
    "병원,의원", "바쁘다,분주하다", "살다,삶,생활", "상하다,다치다,부상,상처,손상", "성공", "수술",
    "쉬다,휴가,휴게,휴식,휴양", "습관,버릇", "시동생", "신기록", "실수", "실패", "아들",
    "아빠,부,부친,아비,아버지", "안과", "안녕,안부", "약", "양치질,양치", "어머니,모친,어미,엄마",
    "여행", "오빠,오라버니", "이기다,승리,승리하다,(경쟁 상대를) 제치다", "입원", "자유,임의,마구,마음껏,마음대로,멋대로,제멋대로,함부로",
    "주무시다,자다,잠들다,잠자다", "죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종", "축구,차다", "취미", "치료", "편찮다,아프다", "할머니,조모", "형,형님", "형제"
]

# 데이터 경로 및 설정
dataset_path = "./AI_WORD/output_npy"
max_frames = 30  # 정규화된 프레임 길이
feature_size = 126  # 두 손 랜드마크 사용 (21 랜드마크 * 3좌표 * 2손)

# 길이 정규화 함수 (30프레임으로 고정)
def normalize_sequence_length(data, target_length, feature_size):
    current_length = len(data)

    if current_length < target_length:
        padding = np.zeros((target_length - current_length, feature_size), dtype='float32')
        return np.vstack((data, padding))
    return data[:target_length]

# 데이터 전처리 함수
def process_gesture_data(actions, dataset_path, max_frames, feature_size):
    X, y = [], []
    total_files = 0
    skipped_files = 0
    gesture_counts = {gesture: 0 for gesture in actions}  # 각 제스처별 데이터 개수 저장

    for idx, gesture in enumerate(actions):
        folder_path = os.path.join(dataset_path, gesture)
        if not os.path.exists(folder_path):
            print(f"⚠️ 경고: {gesture} 폴더 없음! 스킵합니다.")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(folder_path, file_name)
                sequence_data = np.load(file_path)

                total_files += 1  # 총 파일 개수 증가

                # 데이터 크기 확인
                if sequence_data.shape[1] not in [63, 126]:
                    print(f"⚠️ 오류: {file_path}의 데이터 형태가 {sequence_data.shape}입니다. 스킵합니다.")
                    skipped_files += 1  # 스킵된 파일 개수 증가
                    continue

                # 데이터 길이 먼저 30 프레임으로 맞춤
                sequence_data = normalize_sequence_length(sequence_data, max_frames, sequence_data.shape[1])

                # 한 손 동작인 경우 오른손 데이터를 0으로 패딩하여 (30, 126) 변환
                if sequence_data.shape[1] == 63:
                    padding = np.zeros((max_frames, 63))  # 오른손 좌표 63개를 0으로 채움
                    sequence_data = np.hstack((sequence_data, padding))  # (30, 126) 형태로 변환

                # 최종 데이터 추가
                X.append(sequence_data)
                y.append(idx)
                gesture_counts[gesture] += 1  # 제스처별 데이터 개수 증가

                # 데이터 증강: 좌우 반전 추가 (X 좌표 반전)
                flipped_data = sequence_data.copy()
                flipped_data[:, :21] *= -1  # 왼손의 X 좌표 반전
                flipped_data[:, 63:84] *= -1  # 오른손의 X 좌표 반전
                X.append(flipped_data)
                y.append(idx)
                gesture_counts[gesture] += 1  # 증강된 데이터 포함

    print("\n📊 데이터 로드 완료")
    print(f"✅ 총 로드된 파일 개수: {total_files}")
    print(f"❌ 스킵된 파일 개수: {skipped_files}\n")

    print("📌 제스처별 데이터 개수:")
    for gesture, count in gesture_counts.items():
        print(f"   {gesture}: {count}개")

    return np.array(X, dtype='float32'), np.array(y)

# 데이터 저장 함수
def save_preprocessed_data(X, y, save_dir="./AI_WORD/processed_dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Label One-Hot Encoding
    y = to_categorical(y, num_classes=len(gestures))

    # 훈련/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 저장
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

    print("\n📊 데이터 저장 완료")
    print(f"✅ 훈련 데이터 크기: {X_train.shape}, 레이블 크기: {y_train.shape}")
    print(f"✅ 검증 데이터 크기: {X_val.shape}, 레이블 크기: {y_val.shape}")

# ✅ 실행
X_data, y_labels = process_gesture_data(gestures, dataset_path, max_frames, feature_size)
save_preprocessed_data(X_data, y_labels)
