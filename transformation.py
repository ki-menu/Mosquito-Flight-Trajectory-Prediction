"""
Data transformation functions for mosquito 3D flight trajectory.
Handles rotation normalization, translation, and feature extraction.
"""
import numpy as np


def get_rotation_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix that aligns the last step direction with the x-axis.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Rotation matrix of shape (3, 3). Returns identity if direction is degenerate.
    """
    if len(coords) < 2:
        return np.eye(3, dtype=np.float32)

    direction = coords[-1] - coords[-2]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    v = direction / norm
    target = np.array([1.0, 0.0, 0.0])

    axis = np.cross(v, target)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # v and target are parallel
        R = np.eye(3) if np.dot(v, target) > 0 else np.diag([-1.0, -1.0, 1.0])
        return R.astype(np.float32)

    axis = axis / axis_norm
    cos_a = np.dot(v, target)
    sin_a = axis_norm

    # Rodrigues' rotation formula
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R.astype(np.float32)


def apply_transformations(sequences: np.ndarray, targets: np.ndarray = None,
                         use_rotation: bool = True, use_delta: bool = False,
                         model_mode: str = 'm2o'):
    """
    Apply rotation, translation (origin shift), and optionally delta conversion.

    Args:
        sequences: (N, T, 3) array of trajectory coordinates.
        targets: (N, 3) for m2o, or (N, 6) for m2m ([t40, t80], NaN where t40 unknown).
        use_rotation: Whether to rotate sequences to align with x-axis.
        use_delta: Whether to convert positions to displacement vectors (deltas).
        model_mode: 'm2o' or 'm2m'.

    Returns:
        transformed_sequences: (N, T', 3)
        transformed_targets: (N, 3) for m2o, (N, 6) for m2m, or None
        last_positions: (N, 3) original last positions before normalization
        rot_mats: (N, 3, 3) rotation matrices used
    """
    N = len(sequences)

    # 1. Rotation matrices
    if use_rotation:
        rot_mats = np.array([get_rotation_matrix(seq) for seq in sequences], dtype=np.float32)
        sequences_rot = np.einsum('ntj,nij->nti', sequences, rot_mats)
    else:
        rot_mats = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
        sequences_rot = sequences

    # 2. Origin shift (last point → origin)
    last_positions_rot = sequences_rot[:, -1, :]
    sequences_norm = sequences_rot - last_positions_rot[:, np.newaxis, :]

    # 3. Delta conversion
    if use_delta:
        sequences_norm = np.diff(sequences_norm, axis=1)

    # 4. Transform targets
    transformed_targets = None
    if targets is not None:
        last_raw = sequences[:, -1, :]  # (N, 3) raw last positions

        if model_mode == 'm2m':
            # targets shape: (N, 6) — first 3 = t40 (may be NaN), last 3 = t80
            t40, t80 = targets[:, :3], targets[:, 3:]

            disp_80 = t80 - last_raw
            trans_t80 = np.einsum('nj,nij->ni', disp_80, rot_mats).astype(np.float32)

            valid = ~np.isnan(t40).any(axis=1)          # (N,) bool
            trans_t40 = np.full((N, 3), np.nan, dtype=np.float32)
            if valid.any():
                disp_40 = t40[valid] - last_raw[valid]
                trans_t40[valid] = np.einsum('nj,nij->ni', disp_40, rot_mats[valid])

            transformed_targets = np.concatenate([trans_t40, trans_t80], axis=1)
        else:
            displacement = targets - last_raw
            transformed_targets = np.einsum('nj,nij->ni', displacement, rot_mats).astype(np.float32)

    return sequences_norm.astype(np.float32), transformed_targets, sequences[:, -1, :].astype(np.float32), rot_mats


def compute_velocity_acceleration(coords: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.diff(coords, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    return velocity, acceleration


def normalize_speed_scale(coords: np.ndarray, dt: float = 1.0) -> np.ndarray:
    speeds = np.linalg.norm(np.diff(coords, axis=0), axis=1) / dt
    scale = np.median(speeds)
    if scale < 1e-8:
        return coords.copy()
    return coords / scale


def remove_speed_outliers(coords: np.ndarray, threshold: float = 3.0, dt: float = 1.0) -> np.ndarray:
    result = coords.copy()
    T = len(coords)
    speeds = np.linalg.norm(np.diff(coords, axis=0), axis=1) / dt
    outlier = np.zeros(T, dtype=bool)
    outlier[1:] = speeds > threshold
    outlier[0] = False
    outlier[-1] = False

    for idx in np.where(outlier)[0]:
        prev = idx - 1
        while prev > 0 and outlier[prev]:
            prev -= 1
        nxt = idx + 1
        while nxt < T - 1 and outlier[nxt]:
            nxt += 1
        alpha = (idx - prev) / (nxt - prev)
        result[idx] = coords[prev] * (1 - alpha) + coords[nxt] * alpha
    return result
