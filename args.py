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
    output_dir = Path('./model')  # 모델 & 결과 모두 여기에 저장

    # 모델 하이퍼파라미터
    input_size = 3    # x, y, z  (delta 모드에서도 feature dim은 동일)
    output_size = 3   # 예측할 x, y, z

    hidden_size = 64
    num_layers = 3
    dropout_rate = 0.1

    # 학습 설정
    batch_size = 128
    epochs = 600

    lr = 1e-3
    min_lr = 1e-6          # LR 하한선 설정
    scheduler_factor = 0.5 # 감쇠 폭 완화 (0.1 -> 0.5)
    patience = 40          
    warmup_epochs = 10     # 초기 Warm-up 에폭 수

    seed = 42
    run_name = "GRU"  # wandb 실행 이름

    # Sub-sequence 증강 길이 범위
    subseq_min_len = 2
    subseq_max_len = 11


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

    parser.add_argument('--model_path', type=str, default=None,
                        help="Specific model path for inference")

    parser.set_defaults(rotate=True)

    return parser.parse_args()


def apply_args(args):
    """Parse된 args를 Config에 반영한다."""
    if args.name:
        Config.run_name = args.name
    Config.use_delta    = (args.input == 'delta')
    Config.use_rotation = args.rotate
    Config.subseq_min_len = args.subseq_min
    Config.subseq_max_len = args.subseq_max

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
