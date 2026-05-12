"""
Dataset definition for Mosquito Flight Trajectory prediction.
Handles data loading, caching, augmentation application, and normalization.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from augmentation import generate_subsequences
from transformation import apply_transformations


class MosquitoDataset(Dataset):
    _cache_dir = Path('./data/.cache')

    def __init__(self, file_paths, labels_df=None, is_train=True, augment_fns=None,
                 use_delta=False, use_rotation=True, subseq_aug=False,
                 subseq_min_len=2, subseq_max_len=11):
        self.is_train = is_train
        self.augment_fns = augment_fns if augment_fns is not None else []
        self.subseq_aug = subseq_aug
        self.subseq_min_len = subseq_min_len
        self.subseq_max_len = subseq_max_len

        # ── 1. raw sequences 로드 (캐시 우선 → 병렬 I/O) ─────────────────
        self._cache_dir.mkdir(exist_ok=True)
        cache_key = f"{len(file_paths)}_{file_paths[0].stem}_{file_paths[-1].stem}"
        cache_file = self._cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            print(f"캐시 로드: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            raw = data['sequences']                    # (N, T, 3)
            self.file_ids = data['file_ids'].tolist()
        else:
            def _load(path):
                return np.loadtxt(str(path), delimiter=',', skiprows=1,
                                  usecols=(1, 2, 3), dtype=np.float32)

            n_workers = min(32, (os.cpu_count() or 1) * 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                seqs = list(tqdm(pool.map(_load, file_paths),
                                 total=len(file_paths), desc="Loading data"))

            self.file_ids = [p.stem for p in file_paths]
            raw = np.stack(seqs)                       # (N, T, 3)
            np.savez(cache_file, sequences=raw, file_ids=np.array(self.file_ids))
            print(f"캐시 저장: {cache_file}")

        # ── 1.5. 라벨 배열 준비 (is_train 인 경우) ────────────────────────
        if is_train and labels_df is not None:
            labels_dict  = labels_df.set_index('id')[['x', 'y', 'z']].T.to_dict('list')
            original_targets = np.array(
                [labels_dict[fid] for fid in self.file_ids], dtype=np.float32
            )
        else:
            original_targets = None

        # ── 2. augmentation (학습 시에만, raw에 적용) ─────────────────────
        if is_train and self.augment_fns:
            aug = []
            for seq in raw:
                for fn in self.augment_fns:
                    seq = fn(seq)
                aug.append(seq)
            raw = np.array(aug, dtype=np.float32)

        # ── 2.5. Sub-sequence Augmentation (Forward & Reverse) ──────────
        if is_train and original_targets is not None and self.subseq_aug:
            raw, target_array = generate_subsequences(
                raw, original_targets,
                min_len=self.subseq_min_len, max_len=self.subseq_max_len
            )
        elif is_train and original_targets is not None:
            target_array = original_targets
        else:
            target_array = None

        # ── 3. Transformation (회전 정규화 + 원점 이동 + Delta) ───────────
        # transformation.py의 apply_transformations를 통해 일괄 처리
        sequences_norm, transformed_targets, last_positions, rot_mats = apply_transformations(
            raw, target_array, use_rotation=use_rotation, use_delta=use_delta
        )

        self.sequences      = list(sequences_norm)
        self.last_positions = list(last_positions)
        self.rot_mats       = list(rot_mats)
        self.targets        = list(transformed_targets) if transformed_targets is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        if self.is_train:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return seq, target
        else:
            last_pos = torch.tensor(self.last_positions[idx], dtype=torch.float32)
            rot_mat = torch.tensor(self.rot_mats[idx], dtype=torch.float32)
            file_id = self.file_ids[idx]
            return seq, last_pos, rot_mat, file_id

