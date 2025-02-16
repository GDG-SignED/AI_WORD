import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import random
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt


# ✅ 모델 로드
model = tf.keras.models.load_model("./model/best_transformer_model.keras")

# ✅ 제스처 리스트
GESTURE_LIST = [
    "가족", "감기", "건강", "검사", "결혼", "고모", "꿈", "남동생", "남편", "낫다",
    "노래", "누나", "다니다", "동생", "머무르다", "모자", "몸살", "병원", "바쁘다", "살다",
    "상하다", "성공", "수술", "쉬다", "습관", "시동생", "신기록", "실수", "실패", "아들",
    "아빠", "안과", "안녕", "약", "양치", "어머니", "여행", "오빠", "이기다", "입원",
    "자유", "주무시다", "죽다", "축구", "취미", "치료", "편찮다", "할머니", "형", "형제"
]


# ✅ MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        # ✅ UI 설정
        self.setWindowTitle("✨ Sign Language Quiz ✨")
        self.setStyleSheet("background-color: #eeeael;")  
        self.showFullScreen()  # 전체 화면 설정

        # ✅ 상단 제목
        self.title_label = QLabel("✨ Sign Language Quiz ✨", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 45px; color: #flac85; font-weight: bold; font-family: 'Times New Roman';")  # 주황색, Times New Roman 적용

        # ✅ 웹캠 화면 표시
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #368f5f; background-color: #ffffff;")
        self.video_label.setFixedSize(640, 480)

        # ✅ 문제 표시
        self.quiz_label = QLabel("", self)
        self.quiz_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.quiz_label.setStyleSheet("font-size: 35px; color: #368f5f; font-family:  'Times New Roman'; font-weight: bold; margin-bottom: 30px; text-align: left;")  

        # ✅ 예측된 단어 & 정확도 표시
        self.prediction_label = QLabel("", self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.prediction_label.setStyleSheet("font-size: 30px; color: #368f5f; font-family: 'Times New Roman';font-weight: bold; margin-bottom: 30px; text-align: left;")

        # ✅ 결과 메시지
        self.result_label = QLabel("🖐 Try to follow the sign shown!", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.result_label.setStyleSheet("font-size: 30px; color: #368f5f; font-weight: bold; font-family: 'Times New Roman'; font-weight: bold; margin-bottom: 30px;text-align: left;")  # 녹색 강조

        # ✅ 남은 시간 표시
        self.timer_label = QLabel("", self)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.timer_label.setStyleSheet("font-size: 27px; color: #cc7fb5; font-family: 'Times New Roman'; font-weight: bold; text-align: left;")

        # ✅ 레이아웃 설정
        main_layout = QVBoxLayout()  # 전체 세로 레이아웃

        # 상단 제목 중앙 정렬
        main_layout.addSpacing(50)  # 제목과 화면 상단 여백
        main_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addSpacing(25)  # 제목과 웹캠 + 텍스트 사이 여백

        # 가운데 웹캠과 텍스트를 나란히 배치, 수직 중앙 정렬
        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 왼쪽에 텍스트 정보 배치
        text_layout = QVBoxLayout()
        text_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quiz_layout = QHBoxLayout()
        quiz_layout.addWidget(self.quiz_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(quiz_layout)

        prediction_layout = QHBoxLayout()
        prediction_layout.addWidget(self.prediction_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(prediction_layout)

        result_layout = QHBoxLayout()
        result_layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(result_layout)

        timer_layout = QHBoxLayout()
        timer_layout.addWidget(self.timer_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(timer_layout)


        # 오른쪽에 웹캠 배치
        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 텍스트와 웹캠 레이아웃 합치기
        center_layout.addLayout(text_layout)
        center_layout.addLayout(webcam_layout)

        # 메인 레이아웃에 추가
        main_layout.addLayout(center_layout)
        main_layout.addSpacing(170)  # 화면 하단과의 여백

        self.setLayout(main_layout)

        # ✅ 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ✅ 문제 제한 시간 타이머 (1초씩 감소)
        self.quiz_timer = QTimer()
        self.quiz_timer.timeout.connect(self.time_up)
        self.time_left = 10  # 시작 시간 10초

        # ✅ 시퀀스 데이터 저장 (30 프레임)
        self.sequence = []
        self.current_quiz = None
        self.correct_count = 0  # 정답 카운트

        # ✅ 퀴즈 진행 중인지 확인하는 플래그 추가
        self.quiz_in_progress = False  # 퀴즈가 진행 중이면 True

        # ✅ 초기화: 남은 제스처 관리
        self.remaining_gestures = GESTURE_LIST.copy()
        random.shuffle(self.remaining_gestures)  # 랜덤 순서 섞기

        # ✅ 웹캠 초기화
        self.cap = None
        self.set_new_quiz()

        # ✅ 웹캠 자동 실행 (Start 버튼 없이 실행)
        self.start_webcam()


    def start_webcam(self):
        """웹캠을 시작하는 함수"""
        self.cap = cv2.VideoCapture(0)
        
        # ✅ 타이머 시작
        self.timer.start(30)  # 30ms마다 프레임 업데이트
        self.time_left = 10  # 10초 타이머 시작
        self.quiz_timer.start(1000)  # 1초마다 감소

    def stop_webcam(self):
        """웹캠을 중지하는 함수"""
        self.timer.stop()
        self.quiz_timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.video_label.clear()
        self.result_label.setText("🖐 Try to follow the sign shown!")
        self.quiz_label.setText("")
        self.timer_label.setText("")
        self.prediction_label.setText("")

    def keyPressEvent(self, event):
        """키보드 입력 처리: 'q'를 누르면 종료"""
        if event.key() == Qt.Key.Key_Q:
            self.close()  # 창 닫기

    def set_new_quiz(self):
        """새로운 퀴즈 출제 (중복 실행 방지)"""
        if self.quiz_in_progress:
            return  # 이미 진행 중이면 종료

        self.quiz_in_progress = True  # 중복 실행 방지

        # ✅ 남은 제스처가 없으면 초기화
        if not self.remaining_gestures:
            self.remaining_gestures = GESTURE_LIST.copy()
            random.shuffle(self.remaining_gestures)

        # ✅ 리스트에서 하나만 꺼내기
        self.current_quiz = self.remaining_gestures.pop(0)

        # ✅ 화면에 남은 문제 표시
        remaining_count = len(self.remaining_gestures)
        total_count = len(GESTURE_LIST)
        self.quiz_label.setText(
            f"📝 Follow this sign: {self.current_quiz} (Remaining: {remaining_count}/{total_count})"
        )

        # ✅ 화면 초기화 (예측 값 삭제 최소화)
        self.result_label.setText("🖐 Try to follow the sign shown!")
        self.sequence = []  # 시퀀스 초기화
        self.time_left = 10  
        self.timer_label.setText(f"⏳ Time left: {self.time_left} seconds")

        # ✅ 2초 후 퀴즈 상태 해제 (예측 성공 후만 해제)
        QTimer.singleShot(3000, self.reset_quiz_status)

    def reset_quiz_status(self):
        """퀴즈 상태 플래그를 해제하는 함수"""
        self.quiz_in_progress = False  # 문제 성공 후만 해제

    def time_up(self):
        """시간 초과 시 동작 - 문제를 변경하지 않고 같은 문제 유지"""
        self.time_left -= 1
        self.timer_label.setText(f"⏳ Time left: {self.time_left} seconds")

        if self.time_left == 0:
            self.result_label.setText("❌ Time's up! Try again.")  # 시간이 초과되면 같은 문제 유지
            self.time_left = 10  # 10초 다시 설정
            self.sequence = []  # 시퀀스 리셋
            self.quiz_timer.start(1000)  # 타이머 다시 시작 (문제 변경 안 함)


    def update_frame(self):
        """웹캠 프레임 캡처 및 예측"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # ✅ OpenCV는 BGR이므로 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ MediaPipe를 사용해 손 랜드마크 추출
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            results = hands.process(frame)

        # ✅ 두 손 데이터 저장 (126차원)
        landmarks = np.zeros((126,), dtype=np.float32)
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i > 1:
                    break
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                landmarks[i * 63:(i + 1) * 63] = hand_data


        # ✅ 시퀀스 데이터 저장 (30 프레임 유지)
        self.sequence.append(landmarks)
        if len(self.sequence) > 30:
            self.sequence = self.sequence[1:]

        # ✅ 예측 실행
        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0)
            predictions = model.predict(input_data)[0]
            max_index = np.argmax(predictions)
            predicted_gesture = GESTURE_LIST[max_index]  # ✅ GESTURE_LIST로 수정
            confidence = predictions[max_index] * 100



            # ✅ 예측된 자모음 & 정확도 표시
            self.prediction_label.setText(f"🎯 Prediction: {predicted_gesture} ({confidence:.2f}%)")

        

            if predicted_gesture == self.current_quiz and confidence > 80:  # ✅ 현재 문제와 비교
                self.result_label.setText(f"✅ Correct! {predicted_gesture} ({confidence:.2f}%)")

                # ✅ 문제를 넘어갈 준비: quiz_in_progress를 즉시 False로 설정
                self.quiz_in_progress = False  

                # ✅ 2초 후 새로운 문제 설정 (기존 코드 유지)
                QTimer.singleShot(2000, self.set_new_quiz)  

            elif confidence > 80:
                self.result_label.setText(f"❌ Incorrect! Try again! ({predicted_gesture}, {confidence:.2f}%)")

                


        # ✅ OpenCV → PyQt 변환 후 화면 출력
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

# ✅ 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())