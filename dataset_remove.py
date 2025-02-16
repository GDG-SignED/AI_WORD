import os

# ✅ 데이터 경로 및 사용자 이름 설정
dataset_path = "./AI_WORD/output_npy"
user_name = "이수연"

# ✅ 단어 리스트 (기존과 동일)
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

# ✅ 삭제할 파일 찾기 및 삭제 실행
deleted_count = 0  # 삭제된 파일 개수 확인

for gesture in gestures:
    folder_path = os.path.join(dataset_path, gesture)
    
    if not os.path.exists(folder_path):
        print(f"폴더 없음: {folder_path} (건너뜀)")
        continue

    # 폴더 내부에서 "이수연"이 포함된 파일만 찾기
    files_to_delete = [f for f in os.listdir(folder_path) if f.endswith('.npy') and user_name in f]

    # 파일 삭제 실행
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"삭제 완료: {file_path}")
        except Exception as e:
            print(f"삭제 실패: {file_path} - {e}")

print(f"\n✅ 총 {deleted_count}개의 잘못된 파일 삭제 완료!")
