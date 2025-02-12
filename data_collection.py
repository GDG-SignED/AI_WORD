import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장 경로 및 제스처 설정
DATA_PATH = 'output_npy'  # 데이터 저장 경로
GESTURES = ['실패']  # 제스처 목록

SEQUENCES = 40  # 제스처당 시퀀스 개수
FRAMES = 30  # 시퀀스당 프레임 개수

# 한글 폰트 설정
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"  # 한글 폰트 파일 경로
font = ImageFont.truetype(FONT_PATH, 32)

# 데이터 저장 디렉토리 생성
os.makedirs(DATA_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    for gesture in GESTURES:
        # 기존 파일 개수 확인
        gesture_path = os.path.join(DATA_PATH, gesture)
        existing_files = len([f for f in os.listdir(gesture_path) if f.endswith('.npy')])

        for sequence in range(existing_files, existing_files + SEQUENCES):
            frames = []
            print(f"Collecting data for {gesture}, sequence {sequence + 1}/{existing_files + SEQUENCES}")

            for frame_num in range(FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR -> RGB 변환 및 처리
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # 랜드마크 추출 및 저장 (양손 처리)
                left_hand = np.zeros(21 * 3)  # 왼손 데이터 (초기값: 0)
                right_hand = np.zeros(21 * 3)  # 오른손 데이터 (초기값: 0)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        handedness = results.multi_handedness[idx].classification[0].label
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                        if handedness == "Left":  # 왼손일 경우
                            left_hand = landmarks
                        elif handedness == "Right":  # 오른손일 경우
                            right_hand = landmarks

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 양손 데이터를 하나로 합쳐 저장 (왼손 + 오른손)
                frames.append(np.concatenate([left_hand, right_hand]))

                # OpenCV 이미지를 Pillow 이미지로 변환 (한글 텍스트 출력용)
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # 현재 제스처와 진행 상태 표시 (한글 출력)
                text = f"제스처: {gesture} | 시퀀스: {sequence + 1}/{existing_files + SEQUENCES} | 프레임: {frame_num + 1}/{FRAMES}"
                draw.text((10, 10), text, font=font, fill=(255, 255, 255))  # 흰색 텍스트

                # Pillow 이미지를 다시 OpenCV 이미지로 변환
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # 화면 출력
                cv2.imshow('Data Collection - Press Q to Quit', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # 시퀀스 데이터 저장 (양손 데이터 포함)
            np.save(os.path.join(DATA_PATH, gesture, f"{gesture}_sequence_{sequence}.npy"), np.array(frames))

cap.release()
cv2.destroyAllWindows()
