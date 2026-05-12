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
                          max_len: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """전체 시퀀스 배열에서 Forward + Reverse 서브시퀀스 증강을 수행한다.

    Args:
        raw: 원본 시퀀스 배열, shape (N, T, 3).
        original_targets: 라벨 배열, shape (N, 3).
        seq_len: GRU에 입력할 고정 시퀀스 길이 (default 11).
        min_len: 생성할 서브시퀀스의 최소 길이 (default 2).
        max_len: 생성할 서브시퀀스의 최대 길이 (default 11).

    Returns:
        Tuple of (augmented_raw, augmented_targets):
            - augmented_raw: shape (M, seq_len, 3)
            - augmented_targets: shape (M, 3)
    """
    T = seq_len  # 원본 시퀀스의 총 타임스텝 수
    aug_raw = []
    aug_targets = []

    # end_idx는 0-based 인덱스로, 시퀀스 내 마지막 포인트를 가리킴
    # subsequence 길이 = end_idx - start_idx + 1
    # min_len, max_len은 이 길이를 제한함
    fwd_end_range = range(min_len - 1, T)  # end_idx: min_len-1 ~ T-1

    for seq_idx, seq in enumerate(raw):
        orig_target = original_targets[seq_idx]

        # 1. Forward sub-sequences (정방향 서브시퀀스)
        for end_idx in fwd_end_range:
            if end_idx == T - 2:
                continue  # +40ms target is unknown (index T-2 = idx 9)

            if end_idx == T - 1:
                target = orig_target
            else:
                target = seq[end_idx + 2]

            # start_idx 범위: subsequence 길이가 min_len~max_len 이 되도록
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
            target = seq[start_idx - 2]

            # end_idx 범위: subsequence 길이가 min_len~max_len 이 되도록
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
