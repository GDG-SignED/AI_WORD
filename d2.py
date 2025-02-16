import os
import numpy as np
import re  # 정규식을 이용해 파일에서 숫자 추출

# ✅ 데이터 저장 경로
dataset_path = "./AI_WORD/output_npy"

# ✅ 기준 사용자 및 비교할 사용자 설정
reference_user = "조예인"  # 비교 기준 (평균값을 계산할 사용자)
target_user = "이수연"  # 검사할 사용자

# ✅ 결과 저장
folder_differences = {}

# ✅ 폴더별로 데이터 처리
for root, _, files in os.walk(dataset_path):
    reference_zero_ratios = []  # 기준 사용자의 0 비율 리스트
    target_files = []  # 비교할 사용자 파일 리스트

    # 🔹 기준 사용자 데이터 & 대상 사용자 데이터 분류
    for file in files:
        if file.endswith(".npy"):
            file_path = os.path.join(root, file)
            data = np.load(file_path)

            zero_ratio = np.mean(data == 0) * 100  # 0 비율 계산 (%)

            if reference_user in file:
                reference_zero_ratios.append(zero_ratio)  # 기준 사용자 0 비율 저장
            elif target_user in file:
                target_files.append((file_path, zero_ratio))  # 비교 대상 사용자 파일 저장

    # 🔹 기준 사용자 데이터가 있어야 비교 가능
    if reference_zero_ratios and target_files:
        reference_mean_zero = np.mean(reference_zero_ratios)  # 기준 사용자 0 비율 평균

        # 🔹 비교 대상 사용자 데이터 처리
        large_diff_files = []
        threshold = 15.0  # ✅ 기준보다 10% 이상 클 때만 출력

        for file_path, target_zero in target_files:
            difference = target_zero - reference_mean_zero  # ✅ 차이 계산 (이수연이 더 클 때만)

            if difference > threshold:  # ✅ 차이가 threshold 이상일 때만 출력
                # 🔹 파일명에서 숫자 부분만 추출 (ex. `_이수연_5.npy` → `5`)
                match = re.search(r"_(\d+)\.npy", file_path)
                if match:
                    file_number = int(match.group(1))
                    large_diff_files.append(f"{file_number}")  # 숫자 + 0 비율 표시 #({target_zero:.1f}%)

        # ✅ 차이 나는 파일이 있으면 저장
        if large_diff_files:
            folder_name = os.path.basename(root)  # 폴더명 추출
            folder_differences[folder_name] = large_diff_files

# ✅ 최종 결과 출력
print("\n✅ 폴더별 데이터 차이 비교 완료!")
for folder, details in folder_differences.items():
    print(f"📂 {folder} 폴더: {', '.join(details)} 파일이 평균보다 0 비율이 많이 큼")
