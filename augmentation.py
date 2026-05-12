"""
Data augmentation functions for mosquito 3D flight trajectory.
Focuses on increasing data variety (reversing, sub-sequencing, etc.).
"""
import numpy as np


def reverse_trajectory(coords: np.ndarray) -> np.ndarray:
    """Augment by reversing the time order of the trajectory.
    Simulates a mosquito travelling the same path in the opposite direction.
    """
    return coords[::-1].copy()


def generate_subsequences(raw: np.ndarray, original_targets: np.ndarray,
                          seq_len: int = 11,
                          min_len: int = 2,
                          max_len: int = 11,
                          model_mode: str = 'm2o') -> tuple[np.ndarray, np.ndarray]:
    """전체 시퀀스 배열에서 Forward + Reverse 서브시퀀스 증강을 수행한다.

    Args:
        raw: 원본 시퀀스 배열, shape (N, T, 3).
        original_targets: 라벨 배열, shape (N, 3).
        seq_len: GRU에 입력할 고정 시퀀스 길이 (default 11).
        min_len: 생성할 서브시퀀스의 최소 길이 (default 2).
        max_len: 생성할 서브시퀀스의 최대 길이 (default 11).
        model_mode: 'm2o' → targets shape (M, 3);
                    'm2m' → targets shape (M, 6) = [target_40ms, target_80ms].
                    전체 시퀀스(end_idx=T-1)의 target_40ms는 NaN.

    Returns:
        Tuple of (augmented_raw, augmented_targets).
    """
    T = seq_len  # 원본 시퀀스의 총 타임스텝 수
    aug_raw = []
    aug_targets = []

    fwd_end_range = range(min_len - 1, T)

    for seq_idx, seq in enumerate(raw):
        orig_target = original_targets[seq_idx]

        # 1. Forward sub-sequences (정방향 서브시퀀스)
        for end_idx in fwd_end_range:
            if end_idx == T - 2:
                continue  # +80ms target is unknown at this end_idx

            if end_idx == T - 1:
                t80 = orig_target
                t40 = np.full(3, np.nan, dtype=np.float32)  # 데이터에 없음
            else:
                t80 = seq[end_idx + 2]
                t40 = seq[end_idx + 1]

            target = np.concatenate([t40, t80]) if model_mode == 'm2m' else t80

            start_min = max(0, end_idx - max_len + 1)
            start_max = end_idx - min_len + 1
            for start_idx in range(start_min, start_max + 1):
                sub_seq = seq[start_idx:end_idx + 1]

                pad_len = seq_len - len(sub_seq)
                if pad_len > 0:
                    padded_seq = np.vstack([np.tile(sub_seq[0], (pad_len, 1)), sub_seq])
                else:
                    padded_seq = sub_seq

                aug_raw.append(padded_seq)
                aug_targets.append(target)

        # 2. Reverse sub-sequences (역방향 서브시퀀스)
        for start_idx in range(min_len, T):
            t80 = seq[start_idx - 2]
            t40 = seq[start_idx - 1]  # 항상 유효 (start_idx >= min_len >= 2)

            target = np.concatenate([t40, t80]) if model_mode == 'm2m' else t80

            end_min = start_idx + min_len - 1
            end_max = min(T - 1, start_idx + max_len - 1)
            for end_idx in range(end_min, end_max + 1):
                sub_seq = seq[start_idx:end_idx + 1][::-1]

                pad_len = seq_len - len(sub_seq)
                if pad_len > 0:
                    padded_seq = np.vstack([np.tile(sub_seq[0], (pad_len, 1)), sub_seq])
                else:
                    padded_seq = sub_seq

                aug_raw.append(padded_seq)
                aug_targets.append(target)

    return np.array(aug_raw, dtype=np.float32), np.array(aug_targets, dtype=np.float32)
