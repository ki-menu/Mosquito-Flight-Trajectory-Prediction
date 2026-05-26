import pandas as pd
import numpy as np
import os
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def constant_velocity_predict(sample_path: Path):
    """basecode.py의 핵심 로직: 마지막 두 지점의 속도를 직선으로 연장"""
    df = pd.read_csv(sample_path)
    # 마지막 지점(0ms)과 그 직전 지점(-40ms) 추출
    prev_xyz = df.loc[df.index[-2], ['x', 'y', 'z']].to_numpy(dtype=float)
    last_xyz = df.loc[df.index[-1], ['x', 'y', 'z']].to_numpy(dtype=float)
    
    # 40ms 간격이므로 80ms 후는 2배의 변화량을 더함
    pred_xyz = last_xyz + 2.0 * (last_xyz - prev_xyz)
    return pred_xyz

def calculate_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def evaluate_baseline():
    # 경로 설정 (현재 파일 기준 상위 폴더의 data)
    current_dir = Path(os.path.abspath(__file__)).parent
    root_dir = current_dir.parent
    labels_path = root_dir / 'data' / 'train_labels.csv'
    train_dir = root_dir / 'data' / 'train'
    
    if not labels_path.exists():
        print(f"Error: {labels_path} 파일을 찾을 수 없습니다.")
        return

    labels_df = pd.read_csv(labels_path)
    errors = []
    
    print(f"Evaluating Baseline (Constant Velocity) on {len(labels_df)} samples...")
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Baseline Predicting"):
        file_id = row['id']
        ground_truth = np.array([row['x'], row['y'], row['z']])
        file_path = train_dir / f"{file_id}.csv"
        
        if not file_path.exists(): continue
            
        # 베이스라인 예측
        pred_xyz = constant_velocity_predict(file_path)
        
        # 오차 계산
        err = calculate_distance(pred_xyz, ground_truth)
        errors.append(err)
        
    errors = np.array(errors)
    
    print("\n" + "="*50)
    print("      Baseline Model Evaluation Results")
    print("="*50)
    print(f"Total Samples: {len(errors)}")
    print(f"Mean Distance Error (MAE): {np.mean(errors):.6f} m")
    print(f"Accuracy (< 0.01m)        : {np.mean(errors < 0.01) * 100:.2f} %")
    print("="*50)

if __name__ == "__main__":
    evaluate_baseline()
