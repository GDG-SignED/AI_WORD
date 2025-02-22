import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# ✅ 모델 로드
model = tf.keras.models.load_model("./model/Sign_ED_best.keras")

# ✅ 수어 단어 리스트
ACTIONS = [
    '안녕,안부', '실수', '살다,삶,생활', '가족,식구,세대,가구', '취미',
    '아빠,부,부친,아비,아버지', '건강,기력,강건하다,튼튼하다', '꿈,포부,꿈꾸다',
    '병원,의원', '어머니,모친,어미,엄마', '노래,음악,가요', '검사',
    '쉬다,휴가,휴게,휴식,휴양', '바쁘다,분주하다', '여행',
    '주무시다,자다,잠들다,잠자다', '고모', '치료',
    '자유,임의,마구,마음껏,마음대로,멋대로,제멋대로,함부로', '다니다',
    '이기다,승리,승리하다,(경쟁 상대를) 제치다', '낫다,치유', '성공',
    '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종', '오빠,오라버니',
    '누나,누님', '머무르다,존재,체류,계시다,묵다', '형,형님',
    '남편,배우자,서방', '축구,차다', '실패', '입원', '양치질,양치',
    '아들', '형제', '몸살', '약', '결혼,혼인,화혼', '남동생',
    '감기', '안과', '습관,버릇', '상하다,다치다,부상,상처,손상',
    '수술', '동생', '모자(관계)', '시동생', '편찮다,아프다',
    '신기록', '할머니,조모'
]

# ✅ MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageTestApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # UI 설정
        self.setWindowTitle("수어 테스트 모드")
        self.setStyleSheet("""
            QWidget {
                background-color: #eeeael;
            }
            QLabel {
                color: #368f5f;
                font-family: 'Times New Roman';
            }
            QPushButton {
                background-color: #368f5f;
                color: white;
                padding: 15px;
                border-radius: 5px;
                font-size: 18px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #2d7a4f;
            }
        """)
        self.showFullScreen()

        # 초기화
        self.current_word_index = 0
        self.sequence = []
        
        # 상단 제목
        self.title_label = QLabel("✨ 수어 테스트 모드 ✨", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 45px; font-weight: bold;")

        # 현재 단어 표시
        self.current_word_label = QLabel("", self)
        self.current_word_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.current_word_label.setStyleSheet("font-size: 35px; font-weight: bold; margin: 20px 0;")

        # 예측 결과 표시
        self.prediction_label = QLabel("", self)
        self.prediction_label.setStyleSheet("font-size: 30px;")

        # 웹캠 화면
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 3px solid #368f5f; background-color: white;")

        # 이전/다음 버튼
        self.prev_button = QPushButton("이전 단어 (←)", self)
        self.next_button = QPushButton("다음 단어 (→)", self)
        
        # 버튼 기능 연결
        self.prev_button.clicked.connect(self.previous_word)
        self.next_button.clicked.connect(self.next_word)

        # 레이아웃 설정
        main_layout = QVBoxLayout()
        main_layout.addSpacing(20)
        main_layout.addWidget(self.title_label)
        main_layout.addSpacing(20)
        
        # 정보 표시 영역
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.current_word_label)
        info_layout.addWidget(self.prediction_label)
        info_layout.addStretch()
        
        # 중앙 레이아웃 (정보 + 웹캠)
        center_layout = QHBoxLayout()
        center_layout.addLayout(info_layout)
        center_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(center_layout)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        main_layout.addLayout(button_layout)
        main_layout.addSpacing(30)

        self.setLayout(main_layout)

        # 웹캠 초기화 및 시작
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 첫 번째 단어 표시
        self.update_current_word()

    def keyPressEvent(self, event):
        """키보드 단축키 처리"""
        if event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_Left:
            self.previous_word()
        elif event.key() == Qt.Key.Key_Right:
            self.next_word()

    def update_current_word(self):
        """현재 테스트 중인 단어 업데이트"""
        current_word = ACTIONS[self.current_word_index]
        total_words = len(ACTIONS)
        self.current_word_label.setText(
            f"📝 현재 단어 ({self.current_word_index + 1}/{total_words}):\n{current_word}"
        )
        self.prediction_label.setText("수어를 시작해주세요! 🖐")

    def previous_word(self):
        """이전 단어로 이동"""
        if self.current_word_index > 0:
            self.current_word_index -= 1
            self.update_current_word()
            self.sequence = []

    def next_word(self):
        """다음 단어로 이동"""
        if self.current_word_index < len(ACTIONS) - 1:
            self.current_word_index += 1
            self.update_current_word()
            self.sequence = []
        
    def update_frame(self):
        """웹캠 프레임 업데이트 및 예측"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 원본 프레임을 인식에 사용
        frame_for_detection = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 손 인식 (원본 프레임 사용)
        with mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:
            results = hands.process(frame_for_detection)

        # 랜드마크 데이터 저장
        landmarks = np.zeros((63,), dtype=np.float32)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_for_detection, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # 시퀀스 데이터 저장
        self.sequence.append(landmarks)
        if len(self.sequence) > 30:
            self.sequence = self.sequence[1:]

        # 예측 실행
        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0)
            predictions = model.predict(input_data)[0]
            max_index = np.argmax(predictions)
            predicted_action = ACTIONS[max_index]
            confidence = predictions[max_index] * 100

            # 현재 단어와 예측 결과 비교
            current_word = ACTIONS[self.current_word_index]
            if predicted_action == current_word and confidence > 80:
                result_color = "#2d7a4f"  # 초록색 (성공)
                result_text = "✅ 정확히 인식되었습니다!"
            else:
                result_color = "#cc7fb5"  # 빨간색 (실패)
                result_text = "❌ 다시 시도해보세요"

            # 예측 결과 표시
            self.prediction_label.setStyleSheet(f"font-size: 30px; color: {result_color};")
            self.prediction_label.setText(
                f"🎯 예측된 단어: {predicted_action}\n"
                f"정확도: {confidence:.1f}%\n"
                f"{result_text}"
            )

        # 디스플레이용 프레임 좌우 반전
        display_frame = cv2.flip(frame_for_detection, 1)

        # OpenCV → PyQt 변환 및 화면 출력 (반전된 프레임 사용)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        """앱 종료 시 정리"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageTestApp()
    window.show()
    sys.exit(app.exec())