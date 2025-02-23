import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import random
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from collections import deque

# ✅ 모델 로드 (필요시 compile=False 옵션 추가)
model = tf.keras.models.load_model("./Trained_model/best_transformer_model.keras", compile=False)

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

# ✅ MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        # ✅ UI 설정
        self.setWindowTitle("✨ Sign Language Quiz ✨")
        self.setStyleSheet("background-color: #eeeae1;")  
        self.showFullScreen()

        # ✅ 웹캠 화면 QLabel 추가
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)

        # ✅ 예측된 단어 & 정확도 표시
        self.prediction_label = QLabel("🎯 Prediction: ?", self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.prediction_label.setStyleSheet("font-size: 30px; color: #368f5f; font-weight: bold;")

        # ✅ 레이아웃 설정
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.prediction_label)
        main_layout.addWidget(self.video_label)
        self.setLayout(main_layout)

        # ✅ 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.sequence = []
        self.prediction_buffer = deque(maxlen=10)
        self.start_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame.copy()

        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            results = hands.process(frame)

        # ✅ 손 감지 확인
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ✅ OpenCV → PyQt 변환 후 QLabel에 표시
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # ✅ 손이 감지되었을 때만 데이터 저장
        if results.multi_hand_landmarks:
            landmarks = np.zeros((30, 126), dtype=np.float32)
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                landmarks[:, i * 63:(i + 1) * 63] = hand_data

            self.sequence.append(landmarks)
            if len(self.sequence) > 30:
                self.sequence = self.sequence[-30:]

            # ✅ 입력 데이터 체크 (디버깅용)
            if len(self.sequence) == 30:
                input_data = np.array(self.sequence)
                print("🧐 Input shape before expansion:", input_data.shape)

                if input_data.shape == (30, 30, 126):
                    input_data = input_data[:, 0, :]  # (30, 126)로 변환

                input_data = np.expand_dims(input_data, axis=0)
                print("✅ Final input shape:", input_data.shape)

                # ✅ 예측 실행
                predictions = model.predict(input_data)[0]
                self.prediction_buffer.append(predictions)

                # ✅ 흔들림 방지 (스무딩 적용)
                smoothed_predictions = np.median(self.prediction_buffer, axis=0)
                max_index = np.argmax(smoothed_predictions)
                predicted_gesture = GESTURE_LIST[max_index]
                confidence = smoothed_predictions[max_index] * 100

                # ✅ 예측 결과 업데이트
                self.prediction_label.setText(f"🎯 Prediction: {predicted_gesture} ({confidence:.2f}%)")
                print(f"🔍 예측 결과: {predicted_gesture} ({confidence:.2f}%)")

# ✅ 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())
