import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 🔹 AI 모델 분석 함수 불러오기 (모델 로드는 video_processing.py에서 수행)
from video_processing import analyze_video  

# ✅ Flask 서버 초기화
app = Flask(__name__)
CORS(app)  # CORS 허용 (모든 도메인에서 요청 가능)

# ✅ 파일 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ 업로드 폴더 생성 (없으면 자동 생성)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """ 비디오 파일 업로드 및 AI 분석 API """
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    # ✅ 고유한 파일명 생성 (UUID 적용)
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    file.save(filepath)  # 업로드된 파일 저장

    # ✅ AI 모델을 사용하여 분석 실행
    try:
        result = analyze_video(filepath)
        return jsonify({'message': '분석 완료', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': f'비디오 분석 중 오류 발생: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """ 서버 상태 확인 API """
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8081)
