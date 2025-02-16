import os
import cv2
import mediapipe as mp
import numpy as np
import time

# ✅ MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ✅ 데이터 저장 경로
dataset_path = "./AI_WORD/output_npy"
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
sequence_count = 40  # ✅ 1 단어당 40개 시퀀스 (0~39)
frame_count = 30     # ✅ 1 시퀀스당 30 프레임
capture_duration = 4  # ✅ 1 시퀀스(30프레임) 촬영 시간 (4초)
frame_interval = capture_duration / frame_count  # ✅ 프레임 간 간격 (4초 / 30프레임 ≈ 0.133초)
user_name = "이수연"
delay_between_gestures = 10  # ✅ 단어 간 대기 시간 (초)

# ✅ 데이터 저장 디렉토리 생성
os.makedirs(dataset_path, exist_ok=True)
for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

# ✅ 손 랜드마크 데이터 캡처 함수 (4초 동안 30프레임 촬영)
def capture_hand_data(cap, hands):
    sequence = []
    start_time = time.time()
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임 읽기 실패")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_hand = []
        right_hand = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label  # 'Left' 또는 'Right'
                hand_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                if handedness == "Left":
                    left_hand = hand_data
                else:
                    right_hand = hand_data

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ✅ 한 손만 감지되면, 나머지 손을 0으로 패딩
        while len(left_hand) < 21:
            left_hand.append([0.0, 0.0, 0.0])
        while len(right_hand) < 21:
            right_hand.append([0.0, 0.0, 0.0])

        # ✅ 왼손 + 오른손 데이터를 합쳐 저장
        landmarks = np.array(left_hand + right_hand).flatten()
        sequence.append(landmarks)

        # ✅ 화면에 진행 상황 출력
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording... ({int(elapsed_time)}s/{capture_duration}s)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)

        # ✅ 프레임 간 간격 조절 (4초 동안 30 프레임 촬영)
        time.sleep(frame_interval)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("사용자 중지. 데이터 수집 종료.")
            return None

    return sequence if len(sequence) == frame_count else None

# ✅ 데이터 수집 루프 (이수연 데이터만 확인 후 부족한 부분만 촬영)
def collect_gesture_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("카메라 연결 성공.")
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        for gesture in gestures:
            gesture_path = os.path.join(dataset_path, gesture)
            
            # ✅ 이수연 데이터만 확인 (파일명 중 "이수연"이 포함된 것만 가져옴)
            existing_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy') and user_name in f]

            # ✅ 이수연 데이터 중 번호만 추출
            existing_numbers = set()
            for f in existing_files:
                try:
                    num = int(f.split('_')[-1].split('.')[0])  # 파일명에서 번호 추출
                    existing_numbers.add(num)
                except ValueError:
                    print(f"⚠️ 경고: {f}에서 숫자 추출 실패 (무시됨)")

            # ✅ 0~39 중 없는 번호 찾기
            missing_numbers = [i for i in range(sequence_count) if i not in existing_numbers]

            if not missing_numbers:
                print(f"{gesture}: 데이터 수집 완료 (총 {len(existing_files)}/{sequence_count})")
                continue

            print(f"{gesture}: 부족한 {len(missing_numbers)}개 데이터 수집 시작.")

            for num in missing_numbers:
                file_name = f"{gesture}_{user_name}_{num}.npy"
                save_path = os.path.join(gesture_path, file_name)

                print(f"{gesture}: sequence {num}/{sequence_count} 촬영 중...")

                frame_list = capture_hand_data(cap, hands)
                if frame_list:
                    np.save(save_path, np.array(frame_list))
                    print(f"{gesture} - {file_name} 데이터 저장 완료!")
                else:
                    print(f"{gesture} - {file_name} 데이터 저장 실패!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # 사용자 중지 후 종료

            print(f"단어 '{gesture}' 촬영 완료. {delay_between_gestures}초 간 휴식...")
            time.sleep(delay_between_gestures)

    cap.release()
    cv2.destroyAllWindows()

# ✅ 실행
if __name__ == "__main__":
    collect_gesture_data()
