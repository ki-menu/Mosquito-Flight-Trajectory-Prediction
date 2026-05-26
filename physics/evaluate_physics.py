import pandas as pd
import numpy as np
import os
import sys

# 현재 파일(evaluate_physics.py)의 부모 폴더(physics)를 경로에 추가하여 modelp 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from modelp import predict_mosquito_position

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def calculate_distance(p1, p2):
    """두 점 사이의 유클리드 거리를 계산합니다."""
    return np.sqrt(np.sum((p1 - p2)**2))

def evaluate_physics_model():
    # 프로젝트 루트 경로 찾기 (physics 폴더의 상위 폴더)
    root_dir = os.path.dirname(current_dir)
    labels_path = os.path.join(root_dir, 'data', 'train_labels.csv')
    train_dir = os.path.join(root_dir, 'data', 'train')
    
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} 파일을 찾을 수 없습니다.")
        return

    # 라벨 데이터 로드
    labels_df = pd.read_csv(labels_path)
    
    kf_errors = []
    poly_errors = []
    
    # 시간 배열 생성 (-400ms ~ 0ms, 40ms 간격, 총 11개 지점)
    time_steps = np.round(np.arange(-0.4, 0.01, 0.04), 3)
    
    print(f"Evaluating {len(labels_df)} samples using logic in /physics...")
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Predicting"):
        file_id = row['id']
        ground_truth = np.array([row['x'], row['y'], row['z']])
        
        file_path = os.path.join(train_dir, f"{file_id}.csv")
        if not os.path.exists(file_path):
            continue
            
        # LiDAR 관측 데이터 로드
        obs_df = pd.read_csv(file_path)
        observations = obs_df[['x', 'y', 'z']].values # Shape: (11, 3)
        
        # 물리 모델 예측
        kf_pred, poly_pred = predict_mosquito_position(time_steps, observations, target_future_time=0.08)
        
        # 오차 계산
        kf_err = calculate_distance(kf_pred, ground_truth)
        poly_err = calculate_distance(poly_pred, ground_truth)
        
        kf_errors.append(kf_err)
        poly_errors.append(poly_err)
        
    kf_errors = np.array(kf_errors)
    poly_errors = np.array(poly_errors)
    
    # 결과 요약
    print("\n" + "="*50)
    print("      Physics Model Evaluation Results")
    print("="*50)
    print(f"Total Samples: {len(kf_errors)}")
    
    print(f"\n[1] Kalman Filter (Linear Projection)")
    print(f"    - Mean Distance Error: {np.mean(kf_errors):.6f} m")
    print(f"    - Accuracy (< 0.01m) : {np.mean(kf_errors < 0.01) * 100:.2f} %")
    
    print(f"\n[2] Polynomial Extrapolation (Curve Fitting)")
    print(f"    - Mean Distance Error: {np.mean(poly_errors):.6f} m")
    print(f"    - Accuracy (< 0.01m) : {np.mean(poly_errors < 0.01) * 100:.2f} %")
    print("="*50)

if __name__ == "__main__":
    evaluate_physics_model()
