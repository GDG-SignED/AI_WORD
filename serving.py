import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용

# 경로 설정 (절대 경로 사용)
UPLOAD_FOLDER = r"C:\FinalHand\uploads"
MODEL_PATH = r"C:\FinalHand\model\Sign_ED_best.keras"  # 🚀 절대 경로로 직접 지정
ALLOWED_EXTENSIONS = {'npy', 'json'}

# 제스처 클래스 정의
actions = [
    '안녕,안부', '실수', '살다,삶,생활', '가족,식구,세대,가구', '취미', '아빠,부,부친,아비,아버지',
    '건강,기력,강건하다,튼튼하다', '꿈,포부,꿈꾸다', '병원,의원', '어머니,모친,어미,엄마', '노래,음악,가요',
    '검사', '쉬다,휴가,휴게,휴식,휴양', '바쁘다,분주하다', '여행', '주무시다,자다,잠들다,잠자다',
    '고모', '치료', '자유', '다니다', '이기다,승리', '낫다,치유', '성공', '죽다', '오빠', '누나', '형',
    '남편', '축구', '실패', '입원', '양치질', '아들', '형제', '몸살', '약', '결혼', '남동생', '감기', 
    '안과', '습관', '상처', '수술', '동생', '모자', '시동생', '편찮다', '신기록', '할머니'
]

# 업로드 폴더 생성
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """파일 확장자 확인 함수"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    """모델 초기화 함수"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ 모델 로딩 성공!")
        return model
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {str(e)}")
        return None

# 모델 로딩
model = init_model()

@app.route('/predict', methods=['POST'])
def predict():
    """예측 요청 처리"""
    if model is None:
        return jsonify({'error': 'AI 모델이 로드되지 않았습니다.'}), 500

    try:
        # JSON 데이터 받기
        if request.is_json:
            data = request.get_json()
            if 'sequence' not in data:
                return jsonify({'error': 'JSON 데이터에 "sequence" 키가 없습니다.'}), 400
            sequence = np.array(data['sequence'])

        # NPY 파일 받기
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                try:
                    sequence = np.load(filepath)
                finally:
                    os.remove(filepath)  # 🔥 파일 삭제 보장
            else:
                return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400
        else:
            return jsonify({'error': '입력 데이터가 없습니다.'}), 400

        # 데이터 전처리
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)

        if sequence.shape[1] != 15:
            return jsonify({'error': '입력 시퀀스 길이가 15프레임이어야 합니다.'}), 400

        # 모델 예측
        prediction = model.predict(sequence)
        predicted_index = np.argmax(prediction[0])
        predicted_action = actions[predicted_index]
        confidence = float(prediction[0][predicted_index])

        return jsonify({
            'predicted_action': predicted_action,
            'confidence': confidence,
            'prediction_vector': prediction[0].tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
