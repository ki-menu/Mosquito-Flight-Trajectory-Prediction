"""
Configuration and argument parsing for Mosquito Flight Trajectory prediction.
"""
import argparse
import random
import os
import numpy as np
import torch
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    data_dir = Path('./data')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    train_labels_path = data_dir / 'train_labels.csv'
    sample_sub_path = data_dir / 'sample_submission.csv'
    output_dir = Path('./result')  # 모델 & 결과 모두 여기에 저장

    # 모델 하이퍼파라미터
    input_size = 3 
    output_size = 3   

    hidden_size = 64
    num_layers = 3
    dropout_rate = 0.2

    # 학습 설정
    batch_size = 64
    epochs = 300

    lr = 1e-3
    min_lr = 1e-6          
    scheduler_factor = 0.5
    patience = 10          
    warmup_epochs = 10     

    seed = 42
    run_name = "GRU"  # wandb 실행 이름

    # Sub-sequence 증강 길이 범위
    subseq_min_len = 2
    subseq_max_len = 11

    # 모델 출력 모드
    # 'm2o': +80ms 단일 예측 (output 3D)
    # 'm2m': +40ms, +80ms 동시 예측 (output 6D, 제출엔 뒤 3D만 사용)
    model_mode = 'm2o'

    # WingLoss 관련 하이퍼파라미터
    wing_w = 0.02                 # WingLoss w 파라미터 (원래 0.03)
    wing_epsilon = 0.005          # WingLoss epsilon 파라미터

    # Hit Loss (3D Gaussian) 관련 하이퍼파라미터
    use_hit_loss = True           # CombinedLoss 사용 여부 (False면 기존 WingLoss만)
    sigma_beam = 0.005            # 레이저 빔 유효 표준편차 (m), 대회 기준 0.01m 기반
    sigma_mosquito = 0.002        # 모기 유효 크기 표준편차 (m)
    transition_start = 1.1        # Curriculum 비활성화 (>1.0 → 항상 고정 가중치 유지)
    transition_end = 1.1          # Curriculum 비활성화
    alpha_min = 1.0               # WingLoss 고정 가중치 (α)
    beta_min = 0.005              # HitLoss 고정 가중치 (β)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mosquito Flight Trajectory GRU Training/Inference"
    )

    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'mps'],
                        help="Device: 'auto' (default), 'cpu', 'gpu', 'mps'")

    parser.add_argument('--name', type=str, default=None,
                        help="Wandb run name (default: from Config)")

    parser.add_argument('--input', type=str, default='delta',
                        choices=['raw', 'delta'],
                        help="Input type: 'raw' (11×3 coords) or "
                             "'delta' (10×3 displacement vectors, default)")
            
    parser.add_argument('--no-rotate', dest='rotate', action='store_false',
                        help="Disable rotation normalization (last-step → +x axis). "
                             "Default: rotation ON")

    parser.add_argument('--subseq-min', type=int, default=2,
                        help="Sub-sequence 증강 최소 길이 (default: 2)")
    parser.add_argument('--subseq-max', type=int, default=11,
                        help="Sub-sequence 증강 최대 길이 (default: 11)")

    parser.add_argument('--model-mode', type=str, default='m2o', choices=['m2o', 'm2m'],
                        help="Model output mode: 'm2o' (+80ms 단일 예측, default) or "
                             "'m2m' (+40ms, +80ms 동시 예측 — 제출엔 +80ms head만 사용)")

    parser.add_argument('--model_path', type=str, default=None,
                        help="Specific model path for inference")

    # WingLoss 관련 인자
    parser.add_argument('--wing-w', type=float, default=None,
                        help="Override wing_w for WingLoss (default: 0.02)")

    # Hit Loss 관련 인자
    parser.add_argument('--no-hit-loss', dest='use_hit_loss', action='store_false',
                        help="Disable Hit Loss (use WingLoss only). Default: Hit Loss ON")
    parser.add_argument('--sigma-beam', type=float, default=None,
                        help="Override sigma_beam for GaussianHitLoss (default: 0.005)")
    parser.add_argument('--sigma-mosquito', type=float, default=None,
                        help="Override sigma_mosquito for GaussianHitLoss (default: 0.002)")
    parser.add_argument('--alpha', type=float, default=None,
                        help="WingLoss 고정 가중치 α (default: 1.0). Curriculum 비활성화됨")
    parser.add_argument('--beta', type=float, default=None,
                        help="GaussianHitLoss 고정 가중치 β (default: 0.005). Curriculum 비활성화됨")

    parser.set_defaults(rotate=True, use_hit_loss=True)

    return parser.parse_args()


def apply_args(args):
    """Parse된 args를 Config에 반영한다."""
    if args.name:
        Config.run_name = args.name
    Config.use_delta    = (args.input == 'delta')
    Config.use_rotation = args.rotate
    Config.subseq_min_len = args.subseq_min
    Config.subseq_max_len = args.subseq_max
    Config.model_mode   = args.model_mode

    # WingLoss 설정 반영
    if hasattr(args, 'wing_w') and args.wing_w is not None:
        Config.wing_w = args.wing_w

    # Hit Loss 설정 반영
    Config.use_hit_loss = args.use_hit_loss
    if args.sigma_beam is not None:
        Config.sigma_beam = args.sigma_beam
    if args.sigma_mosquito is not None:
        Config.sigma_mosquito = args.sigma_mosquito
    if args.alpha is not None:
        Config.alpha_min = args.alpha
    if args.beta is not None:
        Config.beta_min = args.beta

    # 디바이스 설정
    if args.device == 'auto':
        if torch.cuda.is_available():
            Config.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            Config.device = torch.device('mps')
        else:
            Config.device = torch.device('cpu')
    elif args.device == 'gpu' and torch.cuda.is_available():
        Config.device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        Config.device = torch.device('mps')
    else:
        if args.device in ['gpu', 'mps']:
            print(f"Warning: {args.device.upper()} not available. Falling back to CPU.")
        Config.device = torch.device('cpu')
