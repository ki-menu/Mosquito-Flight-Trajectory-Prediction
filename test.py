"""
Inference / Prediction for Mosquito Flight Trajectory GRU model.
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from args import Config, parse_args, apply_args
from model import MosquitoGRU
from dataset import MosquitoDataset


def inference(model_path=None):
    """추론을 수행하고 submission CSV를 model/ 디렉토리에 저장한다."""
    # 모델 초기화
    model = MosquitoGRU(
        input_size=Config.input_size, 
        hidden_size=Config.hidden_size, 
        num_layers=Config.num_layers,
        output_size=Config.output_size,
        dropout_rate=Config.dropout_rate
    ).to(Config.device)
    
    # 모델 경로 결정
    if model_path is None:
        # model 폴더에서 가장 성능이 좋은(Dist가 낮은) 파일 검색
        save_files = list(Config.output_dir.glob('gru_*_*.pth'))
        if not save_files:
            raise FileNotFoundError("학습된 모델 파일을 찾을 수 없습니다. (model/gru_*.pth)")
        
        # 파일명 형식: gru_{dist}_{epoch}.pth -> dist(두 번째 요소)가 가장 작은 것 선택
        def get_dist(path):
            try:
                return float(path.stem.split('_')[1])
            except:
                return float('inf')
        
        model_path = sorted(save_files, key=get_dist)[0]
        
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test 데이터셋 로드
    test_files = sorted(list(Config.test_dir.glob('TEST_*.csv')))
    test_dataset = MosquitoDataset(test_files, is_train=False,
                                   use_delta=Config.use_delta,
                                   use_rotation=Config.use_rotation)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for seq, last_pos, rot_mat, file_ids in test_loader:
            seq = seq.to(Config.device)

            # 회전 공간에서 변위 예측
            pred_rotated = model(seq).cpu()  # (B, 3)

            # 역회전: pred @ R  (학습 시 target @ R.T 로 변환했으므로 R.T의 역행렬 = R)
            pred_displacement = torch.bmm(
                pred_rotated.unsqueeze(1), rot_mat
            ).squeeze(1).numpy()  # (B, 3)

            # 최종 예측 위치 = 0ms 위치 + 역회전된 변위
            last_pos_np = last_pos.numpy()
            pred_pos = last_pos_np + pred_displacement
            
            for i in range(len(file_ids)):
                predictions.append({
                    'id': file_ids[i],
                    'x': pred_pos[i, 0],
                    'y': pred_pos[i, 1],
                    'z': pred_pos[i, 2]
                })
                
    # 제출 형식에 맞게 병합 후 저장 (model/ 디렉토리에 저장)
    pred_df = pd.DataFrame(predictions)
    sample_sub = pd.read_csv(Config.sample_sub_path)
    submission = sample_sub[['id']].merge(pred_df, on='id', how='left')
    
    Config.output_dir.mkdir(exist_ok=True)
    # 모델 파일명에서 dist 정보 추출하여 submission 파일명에 반영
    model_stem = Path(model_path).stem
    sub_path = Config.output_dir / f'submission_{model_stem}.csv'
        
    submission.to_csv(sub_path, index=False)
    print(f"Inference complete. Saved to {sub_path}")


if __name__ == '__main__':
    args = parse_args()
    apply_args(args)

    print("--- Starting Inference ---")
    inference(args.model_path)
