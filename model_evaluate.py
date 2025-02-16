import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')  # Windows
plt.rc('axes', unicode_minus=False)

# 🔹 F1 Score metric 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    F1score = 2 * precision * recall / (precision + recall + 1e-7)
    return F1score

# 🔹 모델 평가 함수
def evaluate_model(model_path, x_val, y_val, actions):
    # ✅ 마스킹 문제 해결 (Numpy 배열 변환)
    x_val = np.array(x_val)

    # 모델 로드
    model = load_model(model_path, custom_objects={'metric_F1score': metric_F1score})
    
    # 모델 요약 출력
    print("\n=== Model Summary ===")
    model.summary()

    # 모델 입력 크기 확인
    print("\n=== 모델 입력 형태 ===")
    print(f"모델 기대 입력: {model.input_shape}")
    print(f"데이터 입력: {x_val.shape}")

    # 입력 크기 조정 (모델이 30개 타임스텝을 기대하는 경우)
    if x_val.shape[1] > 30:
        print(f"⚠ 입력 크기 조정: {x_val.shape} → ({x_val.shape[0]}, 30, {x_val.shape[2]})")
        x_val = x_val[:, :30, :]  # 첫 30개 타임스텝만 사용

    # 예측 수행
    y_pred = model.predict(x_val)

    # 원-핫 인코딩된 결과를 클래스 인덱스로 변환
    y_val_classes = np.argmax(y_val, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification Report 생성
    report = classification_report(y_val_classes, y_pred_classes, target_names=actions, output_dict=True)

    # DataFrame으로 변환하여 보기 좋게 출력
    df_report = pd.DataFrame(report).transpose()
    print("\n=== Classification Report ===")
    print(df_report.round(3))

    # Confusion Matrix 생성 및 시각화
    cm = confusion_matrix(y_val_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.xlabel('예측값')
    plt.ylabel('실제값')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 클래스별 성능 분석
    class_performance = {}
    for i, action in enumerate(actions):
        mask = (y_val_classes == i)
        class_acc = np.mean(y_pred_classes[mask] == i) if np.sum(mask) > 0 else 0.0
        class_performance[action] = {
            'accuracy': class_acc,
            'samples': np.sum(mask),
            'correct_predictions': np.sum(y_pred_classes[mask] == i)
        }

    # 성능이 낮은 클래스 식별 (예: 정확도 95% 미만)
    poor_performing = {k: v for k, v in class_performance.items() if v['accuracy'] < 0.95}

    if poor_performing:
        print("\n=== 성능이 낮은 클래스 ===")
        for action, stats in poor_performing.items():
            print(f"{action}:")
            print(f"  정확도: {stats['accuracy']*100:.2f}%")
            print(f"  샘플 수: {stats['samples']}")
            print(f"  정확한 예측: {stats['correct_predictions']}")

    # 전체 모델 성능 메트릭
    test_loss, test_acc = model.evaluate(x_val, y_val, verbose=0)
    print("\n=== 전체 모델 성능 ===")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    return df_report, class_performance

# 🔹 데이터 로드 및 평가 실행
if __name__ == "__main__":
    # 제공된 데이터셋 불러오기
    x_val = np.load('./dataset/x_val.npy')
    y_val = np.load('./dataset/y_val.npy')

    # 🔹 새로운 제스처 클래스 리스트
    actions = [
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
        '아들', '형제', '몸살', '약', '결혼,혼인,화혼',
        '남동생', '감기', '안과', '습관,버릇',
        '상하다,다치다,부상,상처,손상', '수술', '동생', '모자(관계)', '시동생',
        '편찮다,아프다', '신기록', '할머니,조모'
    ]

    # 모델 평가 함수 실행
    report, performance = evaluate_model('./model/Sign_ED_best.keras', x_val, y_val, actions)
