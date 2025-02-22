import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Flask 애플리케이션 생성
app = Flask(__name__)
CORS(app)

def normalize_landmarks(hand_data):
    """1차원 손 데이터 정규화 (63차원 입력)"""
    if hand_data.size != 63 or np.all(hand_data == 0):
        return np.zeros_like(hand_data)

    landmarks = hand_data.reshape(21, 3)
    wrist = landmarks[0]
    palm_vector = landmarks[9] - wrist
    scale_factor = np.linalg.norm(palm_vector)

    if scale_factor < 1e-7:
        return hand_data

    normalized = (landmarks - wrist) / scale_factor
    return normalized.flatten()


# 숫자 클래스 -> 의미 있는 단어 매핑
LABEL_MAPPING = {
    0: '가족,식구,세대,가구',
    1: '건강,기력,강건하다,튼튼하다',
    2: '꿈,포부,꿈꾸다',
    3: '낫다,치유',
    4: '누나,누님',
    5: '다니다',
    6: '동생',
    7: '병원,의원',
    8: '살다,삶,생활',
    9: '수술',
    10: '실패',
    11: '아들',
    12: '아빠,부,부친,아비,아버지',
    13: '안녕,안부',
    14: '양치질,양치',
    15: '어머니,모친,어미,엄마',
    16: '자유,임의,마구,마음껏,마음대로,멋대로,제멋대로,함부로',
    17: '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종',
    18: '할머니,조모',
    19: '형,형님'
}
# 모델 클래스 정의
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.input_processor = nn.Sequential(
            nn.Unflatten(2, (21, 6)),
            nn.Dropout(0.1)
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 3))
        )
        self.temporal_encoder = nn.GRU(
            input_size=16 * 5 * 3,
            hidden_size=32,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.input_processor(x).unsqueeze(2)
        batch_size, timesteps = x.size(0), x.size(1)
        x = x.view(-1, 1, 21, 6)
        features = self.cnn(x).view(batch_size, timesteps, -1)
        _, h_n = self.temporal_encoder(features)
        last_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(last_output)


# 모델 로드
CLASSES = 20
MODEL_PATH = 'model/customized_480_best_model.pth'
model = SignLanguageModel(CLASSES)
model.load_state_dict(torch.load(MODEL_PATH,  map_location=torch.device("cpu")))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(data["keypoints"])
        if "keypoints" not in data:
            return jsonify({"error": "Missing keypoints data"}), 400

        keypoints = np.array(data["keypoints"]).astype(np.float32)  # (30,126) 형태
        if keypoints.shape != (30, 126):
            return jsonify({"error": "Invalid input shape. Expected (30, 126)"}), 400

        # 손 데이터 정규화 적용
        normalized_keypoints = []
        for frame in keypoints:
            norm_hands = []
            for i in range(0, 126, 63):  # 각 손의 좌표를 분리하여 정규화
                norm_hands.append(normalize_landmarks(frame[i:i + 63]))
            normalized_keypoints.append(np.concatenate(norm_hands))  # 다시 합치기

        normalized_keypoints = np.array(normalized_keypoints)

        # PyTorch 텐서 변환 및 모델 예측
        input_tensor = torch.tensor(normalized_keypoints).unsqueeze(0)  # (1,30,126)
        with torch.no_grad():
            output = model(input_tensor)
            prediction_idx = torch.argmax(output, dim=1).item()

        # 숫자 클래스 -> 의미 있는 단어 변환
        prediction_label = LABEL_MAPPING.get(prediction_idx, "Unknown")

        return jsonify({"prediction": prediction_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인을 위한 헬스 체크 API."""
    return jsonify({'status': 'OK'}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
