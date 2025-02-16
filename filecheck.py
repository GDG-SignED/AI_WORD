import numpy as np
import os

dataset_path = "./AI_WORD/output_npy"  # 네 데이터 경로

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):  # 폴더인지 확인
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                file_path = os.path.join(folder_path, file)
                data = np.load(file_path)

                # 30프레임이 아닌 경우, 실제 프레임 개수 출력
                if data.shape[0] != 30:
                    print(f"⚠️ {file_path} - 데이터 shape: {data.shape} (프레임 개수: {data.shape[0]})")
