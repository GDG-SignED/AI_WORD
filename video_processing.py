import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ✅ MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ✅ AI 모델 로드 (손동작 예측 Transformer 모델)
MODEL_PATH = "./Trained_model/trained_model/best_transformer_model.keras"
print("🔹 모델 로드 중...")
model = load_model(MODEL_PATH)
print("✅ 모델 로드 완료!")

# ✅ 수어 제스처 리스트 (모델이 학습한 클래스 순서)
GESTURE_LIST = [
    '가족,식구,세대,가구', '감기', '건강,기력,강건하다,튼튼하다', '검사', '결혼,혼인,화혼', 
    '꿈,포부,꿈꾸다', '남동생', '남편,배우자,서방', '낫다,치유', '살다,삶,생활', 
    '쉬다,휴가,휴게,휴식,휴양', '습관,버릇', '신기록', '안녕,안부', '약', 
    '양치질,양치', '어머니,모친,어미,엄마', '여행', '오빠,오라버니', '취미'
]

def extract_hand_landmarks(frame, hands):
    """ 비디오 프레임에서 손 랜드마크 (126차원) 추출 """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    landmarks = np.zeros(126)  # 기본값 (손이 감지되지 않으면 0으로 채움)
    if results.multi_hand_landmarks:
        temp_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                temp_landmarks.extend([lm.x, lm.y, lm.z])
        
        temp_landmarks = np.array(temp_landmarks[:126])  # 최대 126차원으로 자르기
        landmarks[:len(temp_landmarks)] = temp_landmarks  # 빈 부분 0으로 채우기

    return landmarks

def analyze_video(video_path):
    """ 비디오를 입력으로 받아 손동작을 예측하는 함수 """
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

    sequence = []  # 30 프레임 데이터를 저장할 배열
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = extract_hand_landmarks(frame, hands)  # 손 랜드마크 추출
        sequence.append(landmarks)
        frame_count += 1

        if frame_count == 30:  # 30 프레임을 수집하면 중단
            break

    cap.release()
    hands.close()

    # 프레임이 부족하면 0으로 채우기
    while len(sequence) < 30:
        sequence.append(np.zeros(126))

    input_data = np.array(sequence, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)  # (1, 30, 126) 형태로 변환

    # ✅ 모델 예측 수행
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_gesture = GESTURE_LIST[predicted_index]
    confidence_score = predictions[0][predicted_index]

    return {
        "predicted_class": predicted_gesture,
        "confidence_score": float(confidence_score)
    }
