import os
import numpy as np

# ✅ 데이터 저장 경로
dataset_path = "./AI_WORD/output_npy"

# ✅ 확인할 사용자 이름
user_name = "이수연"

# ✅ 모든 npy 파일 검사
zero_files = []
nonzero_files = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".npy") and user_name in file:
            file_path = os.path.join(root, file)
            data = np.load(file_path)

            # ✅ 모든 값이 0인지 확인
            zero_ratio = np.mean(data == 0) * 100  # 0의 비율 (%)
            is_zero_only = np.all(data == 0)  # 100% 0인지 체크

            # ✅ 출력
            print(f"\n📂 파일: {file_path}")
            # print(f"➡️ 데이터 크기: {data.shape}")
            # print(f"➡️ 첫 번째 프레임 샘플:\n{data[0]}")
            print(f"➡️ 전체 데이터 중 0 비율: {zero_ratio:.2f}%")
            
            # ✅ 분류
            if is_zero_only:
                zero_files.append(file_path)
            else:
                nonzero_files.append(file_path)

# ✅ 결과 출력
print("\n✅ 검사 완료!")
print(f"❌ {len(zero_files)}개의 파일이 0으로만 채워져 있음.")
print(f"✅ {len(nonzero_files)}개의 파일은 정상 데이터 포함.")

if zero_files:
    print("\n⚠️ 0으로만 채워진 파일 목록:")
    for f in zero_files[:10]:  # 너무 많으면 10개만 출력
        print(f)

# ✅ 전체 데이터 개수 출력
print(f"\n📂 총 npy 파일 개수: {len(zero_files) + len(nonzero_files)}")
